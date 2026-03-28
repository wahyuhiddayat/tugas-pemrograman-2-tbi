import math

from index import InvertedIndexReader
from preprocessing import preprocess


def _cosine_sim(vec1, vec2):
    """
    Menghitung cosine similarity antara dua vektor TF-IDF sparse (dict).

    Parameters
    ----------
    vec1, vec2 : dict[str, float]
        Vektor TF-IDF sparse.

    Returns
    -------
    float
        Cosine similarity dalam rentang [0, 1].
    """
    common = set(vec1) & set(vec2)
    if not common:
        return 0.0
    dot = sum(vec1[t] * vec2[t] for t in common)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return dot / (norm1 * norm2)


class AdaptiveReranker:
    """
    Adaptive Re-ranking berbasis Corpus Graph.

    Terinspirasi dari paper "Adaptive Re-Ranking with a Corpus Graph"
    (MacAvaney et al., 2022) yang digunakan di PyTerrier Adaptive.

    Ide utama: dokumen yang mirip (berdasarkan konten) dengan dokumen
    yang sudah diketahui relevan (high BM25 score) cenderung juga relevan.
    Dengan mempropagasikan skor melalui graph kemiripan antar dokumen,
    kita dapat memperbaiki ranking awal dari BM25.

    Algoritma:
    1. Ambil initial_k kandidat teratas dari BM25.
    2. Baca setiap dokumen kandidat, hitung vektor TF-IDF sparse.
    3. Hitung cosine similarity antar semua pasangan dokumen, bentuk corpus graph
       (edge hanya dibuat jika similarity > graph_threshold).
    4. Normalisasi skor BM25 ke [0, 1].
    5. Propagasi skor melalui graph (satu langkah message passing):
         score_new[d] = (1 - alpha) * score_bm25[d] + alpha * Σ_{d'} sim(d, d') * score_bm25[d'] / (|N(d)| + 1)
    6. Re-rank berdasarkan score_new, kembalikan top_k.

    Attributes
    ----------
    bsbi_index : BSBIIndex
        Instance BSBIIndex yang sudah di-load.
    alpha : float
        Bobot untuk kontribusi graph (default 0.4).
        0 = pure BM25, 1 = pure graph propagation.
    initial_k : int
        Jumlah kandidat awal dari BM25 (default 50).
    graph_threshold : float
        Batas minimum cosine similarity untuk membuat edge di corpus graph
        (default 0.1). Nilai lebih tinggi = graph lebih sparse.
    """

    def __init__(self, bsbi_index, alpha=0.97, initial_k=100, graph_threshold=0.05):
        """
        Inisialisasi AdaptiveRetriever.

        Parameters
        ----------
        bsbi_index : BSBIIndex
            Instance BSBIIndex yang sudah di-load.
        alpha : float
            Bobot kontribusi graph neighbor (0 = BM25 saja, 1 = graph saja).
        initial_k : int
            Jumlah kandidat BM25 yang diproses sebelum re-ranking.
        graph_threshold : float
            Minimum cosine similarity untuk membuat edge di corpus graph.
        """
        self.bsbi_index = bsbi_index
        self.alpha = alpha
        self.initial_k = initial_k
        self.graph_threshold = graph_threshold

    def _read_document(self, doc_path):
        """
        Membaca dan mempreproses sebuah dokumen dari disk.

        Parameters
        ----------
        doc_path : str
            Path ke file dokumen.

        Returns
        -------
        list[str]
            List of stemmed tokens.
        """
        try:
            with open(doc_path, "r", encoding="utf-8", errors="surrogateescape") as f:
                return preprocess(f.read())
        except OSError:
            return []

    def _build_tfidf_vectors(self, doc_paths, N, postings_dict):
        """
        Membangun vektor TF-IDF sparse untuk setiap dokumen kandidat.

        Parameters
        ----------
        doc_paths : list[str]
            Daftar path dokumen.
        N : int
            Jumlah total dokumen di koleksi.
        postings_dict : dict
            postings_dict dari InvertedIndexReader.

        Returns
        -------
        list[dict[str, float]]
            List vektor TF-IDF, satu per dokumen.
        """
        str_to_id = self.bsbi_index.term_id_map.str_to_id
        vectors = []
        for path in doc_paths:
            tokens = self._read_document(path)
            tf_dict = {}
            for t in tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            vec = {}
            for term, tf in tf_dict.items():
                term_id = str_to_id.get(term)
                if term_id is not None and term_id in postings_dict:
                    df = postings_dict[term_id][1]
                    if df > 0:
                        idf = math.log(N / df)
                        vec[term] = (1 + math.log(tf)) * idf
            vectors.append(vec)
        return vectors

    def _build_corpus_graph(self, vectors):
        """
        Membangun corpus graph dari vektor TF-IDF dokumen kandidat.

        Edge dibuat antara dua dokumen jika cosine similarity-nya melebihi
        graph_threshold. Hasilnya adalah adjacency list dengan bobot.

        Parameters
        ----------
        vectors : list[dict[str, float]]
            Vektor TF-IDF dokumen kandidat.

        Returns
        -------
        list[list[tuple[int, float]]]
            adjacency[i] = [(j, sim), ...] untuk setiap dokumen i.
        """
        n = len(vectors)
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = _cosine_sim(vectors[i], vectors[j])
                if sim >= self.graph_threshold:
                    adjacency[i].append((j, sim))
                    adjacency[j].append((i, sim))
        return adjacency

    def _propagate_scores(self, bm25_scores, adjacency):
        """
        Propagasi skor BM25 melalui corpus graph (satu langkah message passing).

        Formula:
          score_new[d] = (1 - alpha) * score_norm[d]
                        + alpha * Σ_{d'} sim(d,d') * score_norm[d'] / (|N(d)| + 1)

        Normalisasi dengan (|N(d)| + 1) mencegah dokumen dengan banyak
        tetangga mendominasi.

        Parameters
        ----------
        bm25_scores : list[float]
            Skor BM25 untuk setiap kandidat (sudah dinormalisasi ke [0, 1]).
        adjacency : list[list[tuple[int, float]]]
            Corpus graph adjacency list.

        Returns
        -------
        list[float]
            Skor setelah propagasi.
        """
        n = len(bm25_scores)
        new_scores = []
        for i in range(n):
            neighbors = adjacency[i]
            if neighbors:
                neighbor_contrib = sum(
                    sim * bm25_scores[j] for j, sim in neighbors
                )
                graph_score = neighbor_contrib / (len(neighbors) + 1)
            else:
                graph_score = 0.0
            new_scores.append(
                (1 - self.alpha) * bm25_scores[i]
                + self.alpha * graph_score
            )
        return new_scores

    def retrieve(self, query, top_k=10):
        """
        Lakukan adaptive re-ranking untuk sebuah query.

        Parameters
        ----------
        query : str
            Query mentah dari pengguna.
        top_k : int
            Jumlah dokumen teratas yang dikembalikan.

        Returns
        -------
        list[tuple[float, str]]
            List of (adaptive_score, doc_path), terurut menurun.
        """
        # Langkah 1: BM25 retrieval awal
        initial_results = self.bsbi_index.retrieve_bm25(
            query, k=self.initial_k
        )
        if not initial_results:
            return []

        bm25_raw = [score for score, _ in initial_results]
        doc_paths = [path for _, path in initial_results]

        # Normalisasi skor BM25 ke [0, 1]
        max_score = max(bm25_raw) if bm25_raw else 1.0
        min_score = min(bm25_raw) if bm25_raw else 0.0
        score_range = max_score - min_score
        if score_range < 1e-10:
            bm25_norm = [1.0] * len(bm25_raw)
        else:
            bm25_norm = [(s - min_score) / score_range for s in bm25_raw]

        # Langkah 2-3: Baca dokumen, bangun corpus graph
        with InvertedIndexReader(self.bsbi_index.index_name, self.bsbi_index.postings_encoding, directory=self.bsbi_index.output_dir) as idx:
            N = len(idx.doc_length)
            vectors = self._build_tfidf_vectors(doc_paths, N, idx.postings_dict)

        adjacency = self._build_corpus_graph(vectors)

        # Langkah 4: Propagasi skor melalui graph
        adaptive_scores = self._propagate_scores(bm25_norm, adjacency)

        # Langkah 5: Re-rank dan kembalikan top_k
        ranked = sorted(
            zip(adaptive_scores, doc_paths),
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[:top_k]


if __name__ == "__main__":
    import re
    from bsbi import BSBIIndex
    from compression import VBEPostings
    from evaluation import load_qrels, ap

    bsbi = BSBIIndex(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    bsbi.load()

    # alpha=0.97, initial_k=400: MAP=0.6085 (+16.4% vs BM25=0.5229)
    # initial_k=100: MAP=0.5734 (+9.7%), faster for interactive use (0.09s/q)
    adaptive = AdaptiveReranker(bsbi, alpha=0.97, initial_k=400, graph_threshold=0.05)

    # Test retrieval
    print("=== Adaptive vs BM25 Retrieval ===\n")
    test_queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]

    for q in test_queries:
        bm25_results = bsbi.retrieve_bm25(q, k=10)
        adaptive_results = adaptive.retrieve(q, top_k=10)
        overlap = len({d for _, d in bm25_results} & {d for _, d in adaptive_results})
        print(f"Query: {q}")
        print(f"  BM25     top-1: {bm25_results[0][1]}")
        print(f"  Adaptive top-1: {adaptive_results[0][1]}")
        print(f"  Top-10 overlap: {overlap}/10\n")

    # Evaluasi MAP: BM25 vs Adaptive
    print("=== Evaluasi MAP: BM25 vs Adaptive ===\n")
    qrels = load_qrels()
    ap_bm25 = 0
    ap_adaptive = 0
    count = 0

    with open("queries.txt") as f:
        for qline in f:
            parts = qline.strip().split()
            qid = parts[0]
            query_text = " ".join(parts[1:])
            num_rel = sum(qrels[qid].values())

            ranking_bm25 = []
            for (score, doc) in bsbi.retrieve_bm25(query_text, k=1000):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking_bm25.append(qrels[qid][did])
            ap_bm25 += ap(ranking_bm25, num_rel)

            ranking_adaptive = []
            for (score, doc) in adaptive.retrieve(query_text, top_k=1000):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking_adaptive.append(qrels[qid][did])
            ap_adaptive += ap(ranking_adaptive, num_rel)

            count += 1

    print(f"  BM25     MAP = {ap_bm25 / count:.4f}")
    print(f"  Adaptive MAP = {ap_adaptive / count:.4f}")
    if ap_adaptive > ap_bm25:
        delta = (ap_adaptive - ap_bm25) / count
        print(f"  Adaptive meningkatkan MAP sebesar +{delta:.4f} "
              f"(+{delta / (ap_bm25 / count) * 100:.1f}%)")
    else:
        print(f"  Adaptive lebih rendah dari BM25 "
              f"(alpha={adaptive.alpha}, threshold={adaptive.graph_threshold})")