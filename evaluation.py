import math
import re
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


######## >>>>> IR metric: DCG

def dcg(ranking):
  """ menghitung search effectiveness metric score dengan
      Discounted Cumulative Gain (DCG)

      DCG mengukur kualitas ranking dengan memberikan bobot lebih besar
      pada dokumen relevan yang muncul di posisi lebih awal.
      Discount menggunakan fungsi logaritmik: gain_i / log2(i + 1)
      di mana i adalah posisi (dimulai dari 1).

      Formula:
          DCG = sum_{i=1}^{n} rel_i / log2(i + 1)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.

      Returns
      -------
      Float
        score DCG
  """
  score = 0.
  for i, rel in enumerate(ranking):
    if rel:
      score += 1.0 / math.log2(i + 2)  # posisi ke-(i+1), discount = log2(i+2)
  return score


######## >>>>> IR metric: NDCG

def ndcg(ranking, num_relevant):
  """ menghitung search effectiveness metric score dengan
      Normalized Discounted Cumulative Gain (NDCG)

      NDCG menormalisasi DCG dengan membaginya dengan IDCG (Ideal DCG),
      yaitu DCG dari ranking sempurna di mana semua dokumen relevan
      ditempatkan di posisi paling atas.

      Formula:
          IDCG = DCG dari ideal ranking (semua relevan di atas)
          NDCG = DCG / IDCG

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.

      num_relevant: int
         total jumlah dokumen relevan yang tersedia di koleksi untuk
         query ini. Digunakan untuk membangun ideal ranking.

      Returns
      -------
      Float
        score NDCG antara 0.0 dan 1.0
  """
  # Ideal ranking: semua dokumen relevan di posisi teratas
  ideal = [1] * min(num_relevant, len(ranking)) + \
          [0] * max(0, len(ranking) - num_relevant)
  idcg = dcg(ideal)
  if idcg == 0:
    return 0.
  return dcg(ranking) / idcg


######## >>>>> IR metric: AP

def ap(ranking, num_relevant):
  """ menghitung search effectiveness metric score dengan
      Average Precision (AP)

      AP mengukur rata-rata precision pada setiap posisi di mana
      dokumen relevan ditemukan. Metrik ini sensitif terhadap
      posisi dokumen relevan dalam hasil pencarian.

      Formula:
          AP = (1 / R) * sum_{k=1}^{n} P(k) * rel(k)

      di mana:
          R     = total jumlah dokumen relevan di koleksi
          P(k)  = precision pada rank k
          rel(k)= 1 jika dokumen di rank k relevan, 0 sebaliknya

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.

      num_relevant: int
         total jumlah dokumen relevan yang tersedia di koleksi untuk
         query ini. Digunakan sebagai pembagi (R) pada formula AP.

      Returns
      -------
      Float
        score AP antara 0.0 dan 1.0
  """
  if num_relevant == 0:
    return 0.
  score = 0.
  num_relevant_so_far = 0
  for i, rel in enumerate(ranking):
    if rel:
      num_relevant_so_far += 1
      score += num_relevant_so_far / (i + 1)  # precision at rank i+1
  return score / num_relevant


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels)
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents.
    evaluasi dilakukan untuk dua metode retrieval: TF-IDF dan BM25,
    dengan empat metrik: RBP, DCG, NDCG, dan AP.
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores_tfidf  = []
    dcg_scores_tfidf  = []
    ndcg_scores_tfidf = []
    ap_scores_tfidf   = []

    rbp_scores_bm25   = []
    dcg_scores_bm25   = []
    ndcg_scores_bm25  = []
    ap_scores_bm25    = []

    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # total dokumen relevan di koleksi untuk query ini
      # (diperlukan untuk NDCG dan AP)
      num_relevant = sum(qrels[qid].values())

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking_tfidf = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking_tfidf.append(qrels[qid][did])
      rbp_scores_tfidf.append(rbp(ranking_tfidf))
      dcg_scores_tfidf.append(dcg(ranking_tfidf))
      ndcg_scores_tfidf.append(ndcg(ranking_tfidf, num_relevant))
      ap_scores_tfidf.append(ap(ranking_tfidf, num_relevant))

      ranking_bm25 = []
      for (score, doc) in BSBI_instance.retrieve_bm25(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking_bm25.append(qrels[qid][did])
      rbp_scores_bm25.append(rbp(ranking_bm25))
      dcg_scores_bm25.append(dcg(ranking_bm25))
      ndcg_scores_bm25.append(ndcg(ranking_bm25, num_relevant))
      ap_scores_bm25.append(ap(ranking_bm25, num_relevant))

  n = len(rbp_scores_tfidf)
  print("Hasil evaluasi TF-IDF terhadap 30 queries")
  print("RBP  score =", sum(rbp_scores_tfidf)  / n)
  print("DCG  score =", sum(dcg_scores_tfidf)  / n)
  print("NDCG score =", sum(ndcg_scores_tfidf) / n)
  print("AP   score =", sum(ap_scores_tfidf)   / n)
  print()
  print("Hasil evaluasi BM25 terhadap 30 queries")
  print("RBP  score =", sum(rbp_scores_bm25)  / n)
  print("DCG  score =", sum(dcg_scores_bm25)  / n)
  print("NDCG score =", sum(ndcg_scores_bm25) / n)
  print("AP   score =", sum(ap_scores_bm25)   / n)

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  # sanity check metrik baru
  # ranking sempurna: semua relevan di atas
  assert dcg([1, 1, 0]) > dcg([0, 1, 1]), "DCG salah: posisi lebih awal harus lebih tinggi"
  assert ndcg([1, 1, 0], 2) == 1.0, "NDCG salah: ranking sempurna harus = 1.0"
  assert ndcg([0, 0, 0], 2) == 0.0, "NDCG salah: tidak ada relevan harus = 0.0"
  assert ap([1, 0, 1], 2) == (1/1 + 2/3) / 2, "AP salah"

  eval(qrels)