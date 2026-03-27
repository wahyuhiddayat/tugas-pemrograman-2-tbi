import os
import math

from preprocessing import preprocess


class RocchioQueryExpansion:
    """
    Implementasi Pseudo-Relevance Feedback (PRF) dengan formula Rocchio
    untuk query expansion.

    PRF mengasumsikan bahwa top-K dokumen dari hasil retrieval awal adalah
    relevan (pseudo-relevant). Dari dokumen-dokumen tersebut, diekstrak
    term-term dengan bobot TF-IDF tertinggi untuk ditambahkan ke query
    asli. Hal ini memperkaya query dengan term-term yang relevan sehingga
    dapat meningkatkan recall dan precision.

    Formula Rocchio (versi modifikasi tanpa komponen negatif):
        q_new = alpha * q_orig + beta * centroid(pseudo-relevant docs)

    Attributes
    ----------
    bsbi_index : BSBIIndex
        Instance BSBIIndex yang sudah terindeks dan di-load.
    alpha : float
        Bobot untuk query asli (default 1.0).
    beta : float
        Bobot untuk centroid dokumen pseudo-relevant (default 0.75).
    num_feedback_docs : int
        Jumlah dokumen teratas yang dianggap pseudo-relevant (default 5).
    num_expansion_terms : int
        Jumlah term baru yang ditambahkan ke query (default 10).
    data_dir : str
        Path ke direktori koleksi dokumen.
    """

    def __init__(self, bsbi_index, alpha=1.0, beta=0.75, num_feedback_docs=5, num_expansion_terms=10):
        self.bsbi_index = bsbi_index
        self.alpha = alpha
        self.beta = beta
        self.num_feedback_docs = num_feedback_docs
        self.num_expansion_terms = num_expansion_terms
        self.data_dir = bsbi_index.data_dir

    def _read_document(self, doc_path):
        """
        Membaca dan mem-preprocess sebuah dokumen dari disk.

        Parameters
        ----------
        doc_path : str
            Path relatif ke dokumen (seperti yang tersimpan di doc_id_map).

        Returns
        -------
        list[str]
            List of stemmed tokens dari dokumen.
        """
        try:
            with open(doc_path, "r", encoding="utf8", errors="surrogateescape") as f:
                return preprocess(f.read())
        except FileNotFoundError:
            return []

    def _compute_tfidf_vector(self, tokens, N):
        """
        Menghitung vektor TF-IDF dari sebuah dokumen (diberikan sebagai list of tokens).

        TF menggunakan log-normalization: 1 + log(tf) jika tf > 0.
        IDF menggunakan: log(N / df).

        Parameters
        ----------
        tokens : list[str]
            List of stemmed tokens.
        N : int
            Jumlah total dokumen di koleksi.

        Returns
        -------
        dict[str, float]
            Mapping term -> bobot TF-IDF.
        """
        from index import InvertedIndexReader

        # Hitung TF
        tf_dict = {}
        for token in tokens:
            tf_dict[token] = tf_dict.get(token, 0) + 1

        # Hitung TF-IDF
        tfidf = {}
        with InvertedIndexReader(self.bsbi_index.index_name,
                                 self.bsbi_index.postings_encoding,
                                 directory=self.bsbi_index.output_dir) as idx:
            for term, tf in tf_dict.items():
                term_id = self.bsbi_index.term_id_map.str_to_id.get(term)
                if term_id is not None and term_id in idx.postings_dict:
                    df = idx.postings_dict[term_id][1]
                    idf = math.log(N / df)
                    tfidf[term] = (1 + math.log(tf)) * idf

        return tfidf

    def expand_query(self, query):
        """
        Melakukan query expansion menggunakan Pseudo-Relevance Feedback (Rocchio).

        Langkah-langkah:
        1. Jalankan BM25 retrieval awal untuk mendapatkan top-K dokumen.
        2. Baca dan preprocess dokumen-dokumen pseudo-relevant tersebut.
        3. Hitung vektor TF-IDF centroid dari dokumen pseudo-relevant.
        4. Terapkan formula Rocchio: q_new = alpha * q_orig + beta * centroid.
        5. Ambil top-M term dengan bobot tertinggi dari q_new yang belum ada
           di query asli sebagai expansion terms.

        Parameters
        ----------
        query : str
            Query mentah dari pengguna.

        Returns
        -------
        tuple[str, list[str]]
            Tuple berisi:
            - Query yang sudah di-expand (string, terms dipisahkan spasi)
            - List of expansion terms yang ditambahkan
        """
        from index import InvertedIndexReader

        # Dapatkan query terms asli (sudah di-stem)
        original_terms = preprocess(query)
        if not original_terms:
            return query, []

        # Retrieval awal dengan BM25
        initial_results = self.bsbi_index.retrieve_bm25(query, k=self.num_feedback_docs)
        if not initial_results:
            return " ".join(original_terms), []

        # Hitung N (jumlah dokumen)
        with InvertedIndexReader(self.bsbi_index.index_name,
                                 self.bsbi_index.postings_encoding,
                                 directory=self.bsbi_index.output_dir) as idx:
            N = len(idx.doc_length)

        # Bangun vektor query asli (bobot = 1 untuk setiap term)
        q_vector = {}
        for term in original_terms:
            q_vector[term] = q_vector.get(term, 0) + 1.0

        # Hitung centroid TF-IDF dari dokumen pseudo-relevant
        centroid = {}
        num_docs = len(initial_results)
        for _, doc_path in initial_results:
            tokens = self._read_document(doc_path)
            doc_tfidf = self._compute_tfidf_vector(tokens, N)
            for term, weight in doc_tfidf.items():
                centroid[term] = centroid.get(term, 0) + weight / num_docs

        # Terapkan formula Rocchio: q_new = alpha * q_orig + beta * centroid
        q_new = {}
        for term, weight in q_vector.items():
            q_new[term] = self.alpha * weight
        for term, weight in centroid.items():
            q_new[term] = q_new.get(term, 0) + self.beta * weight

        # Pilih expansion terms: top-M term baru (belum ada di query asli)
        original_set = set(original_terms)
        candidates = [(term, weight) for term, weight in q_new.items() if term not in original_set]
        candidates.sort(key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, _ in candidates[:self.num_expansion_terms]]

        expanded_query = " ".join(original_terms + expansion_terms)
        return expanded_query, expansion_terms

    def retrieve_with_prf(self, query, k=10):
        """
        Melakukan retrieval dengan query expansion (PRF + Rocchio),
        lalu menjalankan BM25 pada query yang sudah di-expand.

        Parameters
        ----------
        query : str
            Query mentah dari pengguna.
        k : int
            Jumlah top dokumen yang dikembalikan.

        Returns
        -------
        tuple[list[tuple[float, str]], str, list[str]]
            Tuple berisi:
            - Hasil retrieval BM25 (list of (score, doc_path))
            - Query yang sudah di-expand
            - List of expansion terms
        """
        expanded_query, expansion_terms = self.expand_query(query)
        results = self.bsbi_index.retrieve_bm25(expanded_query, k=k)
        return results, expanded_query, expansion_terms


if __name__ == "__main__":
    from bsbi import BSBIIndex
    from compression import VBEPostings

    bsbi = BSBIIndex(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    bsbi.load()

    prf = RocchioQueryExpansion(bsbi, alpha=1.0, beta=0.75, num_feedback_docs=5, num_expansion_terms=10)

    # Test query expansion
    print("=== Query Expansion (Rocchio PRF) ===\n")
    test_queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]

    for q in test_queries:
        expanded, expansion_terms = prf.expand_query(q)
        original_stems = " ".join(preprocess(q))
        print(f"Query: {q}")
        print(f"  Original stems:   {original_stems}")
        print(f"  Expansion terms:  {expansion_terms}")
        print(f"  Expanded query:   {expanded}")
        print()

    # Bandingkan hasil retrieval: BM25 biasa vs BM25 + PRF
    print("=== Perbandingan Retrieval: BM25 vs BM25+PRF ===\n")
    for q in test_queries:
        bm25_results = bsbi.retrieve_bm25(q, k=10)
        prf_results, expanded, expansion = prf.retrieve_with_prf(q, k=10)

        bm25_docs = [doc for _, doc in bm25_results]
        prf_docs = [doc for _, doc in prf_results]

        # Hitung overlap
        overlap = len(set(bm25_docs) & set(prf_docs))

        print(f"Query: {q}")
        print(f"  BM25 top-1:     {bm25_results[0][1]} (score={bm25_results[0][0]:.3f})")
        print(f"  BM25+PRF top-1: {prf_results[0][1]} (score={prf_results[0][0]:.3f})")
        print(f"  Top-10 overlap: {overlap}/10")
        print(f"  Expansion: +{len(expansion)} terms")
        print()

    # Evaluasi PRF terhadap 30 queries (sama seperti evaluation.py)
    print("=== Evaluasi BM25+PRF terhadap 30 queries ===\n")
    import re
    from evaluation import load_qrels, ap

    qrels = load_qrels()

    ap_bm25 = 0
    ap_prf = 0
    count = 0
    with open("queries.txt") as f:
        for qline in f:
            parts = qline.strip().split()
            qid = parts[0]
            query_text = " ".join(parts[1:])
            num_rel = sum(qrels[qid].values())

            # BM25 biasa
            ranking_bm25 = []
            for (score, doc) in bsbi.retrieve_bm25(query_text, k=1000):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking_bm25.append(qrels[qid][did])
            ap_bm25 += ap(ranking_bm25, num_rel)

            # BM25 + PRF
            prf_results, _, _ = prf.retrieve_with_prf(query_text, k=1000)
            ranking_prf = []
            for (score, doc) in prf_results:
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking_prf.append(qrels[qid][did])
            ap_prf += ap(ranking_prf, num_rel)

            count += 1

    print(f"  BM25     MAP = {ap_bm25 / count:.4f}")
    print(f"  BM25+PRF MAP = {ap_prf / count:.4f}")
    if ap_prf > ap_bm25:
        print(f"  PRF meningkatkan MAP sebesar +{(ap_prf - ap_bm25) / count:.4f} "
              f"(+{(ap_prf - ap_bm25) / ap_bm25 * 100:.1f}%)")
    else:
        print(f"  PRF menurunkan MAP sebesar {(ap_prf - ap_bm25) / count:.4f}")