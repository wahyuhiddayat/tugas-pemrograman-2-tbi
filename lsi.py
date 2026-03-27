import os
import math
import pickle
import numpy as np

from preprocessing import preprocess


class LSIRetriever:
    """
    Implementasi Latent Semantic Indexing (LSI) menggunakan Truncated SVD.

    LSI mengatasi kelemahan TF-IDF dan BM25 dengan memetakan terms dan
    dokumen ke ruang vektor laten berdimensi rendah. Dalam ruang laten ini,
    sinonim dan term-term yang sering muncul bersama memiliki representasi
    yang dekat, sehingga bisa menangkap hubungan semantik antar kata.

    Proses:
    1. Bangun Term-Document Matrix A (|V| x N) dengan bobot TF-IDF.
    2. Terapkan Truncated SVD: A ≈ U_k · Σ_k · V_k^T
       - U_k : matriks term-topic (|V| x k)
       - Σ_k : nilai singular teratas (k x k, diagonal)
       - V_k : matriks document-topic (N x k)
    3. Proyeksi query ke ruang laten: q_k = Σ_k^{-1} · U_k^T · q_tfidf
    4. Hitung cosine similarity antara q_k dan setiap baris V_k
       untuk mendapatkan ranking dokumen.

    Catatan:
    - numpy digunakan untuk operasi matriks dan SVD.
    - Model (U_k, sigma_k, Vt_k, doc_ids) disimpan ke disk agar
      tidak perlu di-fit ulang setiap kali retrieval.

    Attributes
    ----------
    bsbi_index : BSBIIndex
        Instance BSBIIndex yang sudah di-load.
    num_components : int
        Jumlah komponen laten (dimensi ruang LSI). Default 100.
    U_k : np.ndarray, shape (|V|, k)
        Matriks term-topic hasil SVD.
    sigma_k : np.ndarray, shape (k,)
        Nilai singular teratas (bukan diagonal matrix, tapi 1-D array).
    Vt_k : np.ndarray, shape (k, N)
        Transpose dari matriks document-topic.
    doc_ids : list[int]
        Daftar doc ID sesuai urutan kolom di matriks.
    term_id_to_row : dict[int, int]
        Mapping termID -> baris di matriks A.
    """

    def __init__(self, bsbi_index, num_components=100):
        self.bsbi_index = bsbi_index
        self.num_components = num_components
        self.U_k = None
        self.sigma_k = None
        self.Vt_k = None
        self.doc_ids = None
        self.term_id_to_row = None

    def build_term_doc_matrix(self):
        """
        Membangun Term-Document Matrix dengan bobot TF-IDF dari inverted index.

        Matriks A berukuran |V| x N di mana:
        - Baris = term (sesuai urutan di index)
        - Kolom = dokumen (sesuai urutan di doc_length)
        - Nilai = TF-IDF weight: (1 + log tf) * log(N / df)

        Returns
        -------
        tuple[np.ndarray, list[int], dict[int, int]]
            (A, doc_ids, term_id_to_row)
        """
        from index import InvertedIndexReader

        with InvertedIndexReader(self.bsbi_index.index_name,
                                 self.bsbi_index.postings_encoding,
                                 directory=self.bsbi_index.output_dir) as idx:
            N = len(idx.doc_length)
            doc_ids = sorted(idx.doc_length.keys())
            doc_id_to_col = {doc_id: col for col, doc_id in enumerate(doc_ids)}

            # Kumpulkan semua term yang ada di postings_dict
            term_ids = sorted(idx.postings_dict.keys())
            term_id_to_row = {term_id: row for row, term_id in enumerate(term_ids)}
            V = len(term_ids)

            print(f"Membangun Term-Document Matrix ({V} terms x {N} docs)...")

            # Bangun matriks sparse-style: kumpulkan (row, col, value)
            rows, cols, values = [], [], []

            idx.reset()
            for term_id, postings_list, tf_list in idx:
                if term_id not in term_id_to_row:
                    continue
                row = term_id_to_row[term_id]
                df = len(postings_list)
                idf = math.log(N / df)
                for doc_id, tf in zip(postings_list, tf_list):
                    if doc_id not in doc_id_to_col:
                        continue
                    col = doc_id_to_col[doc_id]
                    tfidf = (1 + math.log(tf)) * idf
                    rows.append(row)
                    cols.append(col)
                    values.append(tfidf)

            A = np.zeros((V, N), dtype=np.float32)
            A[rows, cols] = values

        return A, doc_ids, term_id_to_row

    def fit(self):
        """
        Melatih model LSI dengan menjalankan Truncated SVD pada Term-Document Matrix.

        Menggunakan numpy.linalg.svd dengan full_matrices=False, lalu
        memotong ke num_components komponen teratas.
        """
        A, doc_ids, term_id_to_row = self.build_term_doc_matrix()

        print(f"Menjalankan SVD (k={self.num_components})...")
        U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

        # Potong ke k komponen teratas
        k = self.num_components
        self.U_k = U[:, :k]         # (|V|, k)
        self.sigma_k = sigma[:k]    # (k,)
        self.Vt_k = Vt[:k, :]       # (k, N)
        self.doc_ids = doc_ids
        self.term_id_to_row = term_id_to_row

        print(f"SVD selesai. Variance explained: "
              f"{(sigma[:k]**2).sum() / (sigma**2).sum() * 100:.1f}%")

    def _project_query(self, query):
        """
        Memproyeksikan query ke ruang laten LSI.

        Formula: q_k = Σ_k^{-1} · U_k^T · q_tfidf

        Vektor query q_tfidf dibangun dengan bobot TF * IDF,
        lalu dinormalisasi, kemudian diproyeksikan ke ruang laten.

        Parameters
        ----------
        query : str
            Query mentah.

        Returns
        -------
        np.ndarray, shape (k,) or None
            Vektor query dalam ruang laten, atau None jika tidak ada term
            yang cocok.
        """
        from index import InvertedIndexReader

        terms = preprocess(query)
        if not terms:
            return None

        with InvertedIndexReader(self.bsbi_index.index_name,
                                 self.bsbi_index.postings_encoding,
                                 directory=self.bsbi_index.output_dir) as idx:
            N = len(idx.doc_length)
            q_vec = np.zeros(len(self.term_id_to_row), dtype=np.float32)

            tf_dict = {}
            for t in terms:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            for term, tf in tf_dict.items():
                term_id = self.bsbi_index.term_id_map.str_to_id.get(term)
                if term_id is None or term_id not in self.term_id_to_row:
                    continue
                if term_id not in idx.postings_dict:
                    continue
                df = idx.postings_dict[term_id][1]
                idf = math.log(N / df)
                row = self.term_id_to_row[term_id]
                q_vec[row] = (1 + math.log(tf)) * idf

        if q_vec.sum() == 0:
            return None

        # q_k = Σ_k^{-1} · U_k^T · q
        q_k = (self.U_k.T @ q_vec) / self.sigma_k # (k,)
        return q_k

    def retrieve(self, query, top_k=10):
        """
        Melakukan retrieval dengan cosine similarity di ruang laten LSI.

        Parameters
        ----------
        query : str
            Query mentah.
        top_k : int
            Jumlah dokumen teratas yang dikembalikan.

        Returns
        -------
        list[tuple[float, str]]
            List of (cosine_similarity, doc_path), terurut menurun.
        """
        if self.U_k is None:
            raise RuntimeError("Model belum di-fit. Panggil fit() atau load() terlebih dahulu.")

        q_k = self._project_query(query)
        if q_k is None:
            return []

        # Kolom Vt_k adalah vektor dokumen (shape k x N), normalisasi L2
        doc_vecs = self.Vt_k.T # (N, k), setiap baris = satu dokumen
        q_norm = q_k / (np.linalg.norm(q_k) + 1e-10)
        doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10
        doc_vecs_norm = doc_vecs / doc_norms

        similarities = doc_vecs_norm @ q_norm # (N,)

        # Ambil top-K
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            doc_path = self.bsbi_index.doc_id_map[doc_id]
            results.append((float(similarities[idx]), doc_path))

        return results

    def save(self, filepath):
        """Menyimpan model LSI (U_k, sigma_k, Vt_k, doc_ids, term_id_to_row) ke disk."""
        model = {
            'U_k': self.U_k,
            'sigma_k': self.sigma_k,
            'Vt_k': self.Vt_k,
            'doc_ids': self.doc_ids,
            'term_id_to_row': self.term_id_to_row,
            'num_components': self.num_components,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model LSI disimpan ke {filepath}")

    def load(self, filepath):
        """Memuat model LSI dari disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        self.U_k = model['U_k']
        self.sigma_k = model['sigma_k']
        self.Vt_k = model['Vt_k']
        self.doc_ids = model['doc_ids']
        self.term_id_to_row = model['term_id_to_row']
        self.num_components = model['num_components']
        print(f"Model LSI dimuat dari {filepath} (k={self.num_components})")


if __name__ == "__main__":
    from bsbi import BSBIIndex
    from compression import VBEPostings
    import re

    bsbi = BSBIIndex(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    bsbi.load()

    lsi_path = os.path.join('index', 'lsi_model.pkl')

    # Fit atau load model
    lsi = LSIRetriever(bsbi, num_components=100)
    if os.path.exists(lsi_path):
        lsi.load(lsi_path)
    else:
        lsi.fit()
        lsi.save(lsi_path)

    # Test retrieval
    print("\n=== Retrieval LSI vs BM25 ===\n")
    test_queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]

    for q in test_queries:
        lsi_results = lsi.retrieve(q, top_k=10)
        bm25_results = bsbi.retrieve_bm25(q, k=10)
        overlap = len({d for _, d in lsi_results} & {d for _, d in bm25_results})
        print(f"Query: {q}")
        print(f"  LSI  top-1: {lsi_results[0][1]} (sim={lsi_results[0][0]:.4f})")
        print(f"  BM25 top-1: {bm25_results[0][1]} (score={bm25_results[0][0]:.3f})")
        print(f"  Top-10 overlap: {overlap}/10")
        print()

    # Evaluasi MAP LSI vs BM25
    print("=== Evaluasi MAP: BM25 vs LSI ===\n")
    from evaluation import load_qrels, ap

    qrels = load_qrels()
    ap_bm25 = 0
    ap_lsi = 0
    count = 0

    with open("queries.txt") as f:
        for qline in f:
            parts = qline.strip().split()
            qid = parts[0]
            query_text = " ".join(parts[1:])
            num_rel = sum(qrels[qid].values())

            # BM25
            ranking_bm25 = []
            for (score, doc) in bsbi.retrieve_bm25(query_text, k=1000):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking_bm25.append(qrels[qid][did])
            ap_bm25 += ap(ranking_bm25, num_rel)

            # LSI
            ranking_lsi = []
            for (sim, doc) in lsi.retrieve(query_text, top_k=1000):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking_lsi.append(qrels[qid][did])
            ap_lsi += ap(ranking_lsi, num_rel)

            count += 1

    print(f"  BM25 MAP = {ap_bm25 / count:.4f}")
    print(f"  LSI  MAP = {ap_lsi / count:.4f}")
    if ap_lsi > ap_bm25:
        print(f"  LSI meningkatkan MAP sebesar +{(ap_lsi - ap_bm25) / count:.4f}")
    else:
        print(f"  LSI lebih rendah dari BM25 sebesar {(ap_bm25 - ap_lsi) / count:.4f}")
        print("  (Normal: LSI unggul untuk synonymy, BM25 unggul untuk exact match)")