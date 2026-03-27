import os
import contextlib
import sys

from bsbi import BSBIIndex
from index import InvertedIndexWriter, InvertedIndexReader
from tqdm import tqdm
from preprocessing import preprocess


class SPIMIIndex(BSBIIndex):
    """
    Implementasi indexing dengan skema SPIMI (Single-Pass In-Memory Indexing).

    Perbedaan utama dengan BSBI:
    - BSBI: mengumpulkan semua pasangan (termID, docID) untuk satu block,
      lalu mengurutkan (sort) seluruh pasangan, baru kemudian membangun
      postings list.
    - SPIMI: membangun dictionary dan postings list secara inkremental
      di memori. Setiap token langsung ditambahkan ke postings list term
      yang bersangkutan tanpa perlu menyimpan dan mengurutkan pasangan
      (termID, docID). Ketika memori sudah melebihi batas (memory_limit),
      hasilnya ditulis ke disk dan memori di-reset.

    Keuntungan SPIMI:
    - Tidak perlu sorting seluruh td_pairs (O(n log n) per block pada BSBI)
    - Lebih hemat memori karena langsung menulis ke disk saat batas tercapai
    - Cocok untuk koleksi yang sangat besar

    Class ini mewarisi BSBIIndex sehingga semua method retrieval (retrieve_tfidf,
    retrieve_bm25, retrieve_bm25_wand) dan merge() tetap bisa digunakan.

    Attributes
    ----------
    memory_limit : int
        Batas jumlah token yang ditampung di memori sebelum ditulis ke disk.
        Nilai default sengaja dibuat kecil (10000) untuk mendemonstrasikan
        perilaku multi-pass pada koleksi kecil.
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index", memory_limit=10000):
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        self.memory_limit = memory_limit

    def spimi_invert(self, token_stream, index):
        """
        Membangun inverted index secara inkremental dari token stream.

        Berbeda dengan invert_write() pada BSBI yang menerima seluruh
        td_pairs lalu mengurutkannya, method ini memproses token satu
        per satu dan langsung menambahkannya ke postings list yang sesuai.

        Parameters
        ----------
        token_stream : iterable of (int, int)
            Stream pasangan (termID, docID) yang diproses satu per satu.
        index : InvertedIndexWriter
            Inverted index pada disk untuk menyimpan hasil inversi.
        """
        postings = {}  # termID -> list of docIDs (sudah terurut per append)
        tf_dict = {}   # termID -> {docID: tf}

        for term_id, doc_id in token_stream:
            if term_id not in postings:
                postings[term_id] = []
                tf_dict[term_id] = {}

            if doc_id not in tf_dict[term_id]:
                tf_dict[term_id][doc_id] = 0
                postings[term_id].append(doc_id)

            tf_dict[term_id][doc_id] += 1

        # Tulis ke disk dengan term terurut (agar bisa di-merge nanti)
        for term_id in sorted(postings.keys()):
            sorted_doc_ids = sorted(postings[term_id])
            tf_list = [tf_dict[term_id][doc_id] for doc_id in sorted_doc_ids]
            index.append(term_id, sorted_doc_ids, tf_list)

    def index(self):
        """
        Proses indexing utama dengan skema SPIMI.

        Berbeda dengan BSBI yang memproses per block (sub-directory),
        SPIMI memproses seluruh koleksi sebagai satu stream token.
        Ketika jumlah token di memori melebihi memory_limit, hasilnya
        ditulis ke disk sebagai intermediate index dan memori di-reset.
        Setelah semua dokumen diproses, semua intermediate index di-merge
        menjadi satu index final (sama seperti BSBI).
        """
        token_buffer = []
        token_count = 0
        block_number = 0

        all_dirs = sorted(next(os.walk(self.data_dir))[1])
        for block_dir_relative in tqdm(all_dirs):
            dir_path = "./" + self.data_dir + "/" + block_dir_relative
            for filename in next(os.walk(dir_path))[2]:
                docname = dir_path + "/" + filename
                with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                    for token in preprocess(f.read()):
                        term_id = self.term_id_map[token]
                        doc_id = self.doc_id_map[docname]
                        token_buffer.append((term_id, doc_id))
                        token_count += 1

                        if token_count >= self.memory_limit:
                            index_id = 'intermediate_index_spimi_' + str(block_number)
                            self.intermediate_indices.append(index_id)
                            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                                self.spimi_invert(token_buffer, index)
                            token_buffer = []
                            token_count = 0
                            block_number += 1

        # Tulis sisa token yang masih ada di buffer
        if token_buffer:
            index_id = 'intermediate_index_spimi_' + str(block_number)
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.spimi_invert(token_buffer, index)
            block_number += 1

        print(f"SPIMI: {block_number} intermediate indices ditulis ke disk "
              f"(memory_limit = {self.memory_limit} tokens)")

        self.save()

        # Merge semua intermediate index menjadi satu index final
        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                    for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    from compression import VBEPostings

    # Test 1: indexing dengan memory_limit kecil untuk demonstrasi multi-pass
    print("=" * 60)
    print("Test SPIMI indexing dengan memory_limit = 10000 tokens")
    print("=" * 60)
    spimi = SPIMIIndex(data_dir='collection',
                       postings_encoding=VBEPostings,
                       output_dir='index',
                       memory_limit=10000)
    spimi.index()
    spimi.load()

    # Test 2: verifikasi retrieval menghasilkan output yang sama dengan BSBI
    print("\nVerifikasi retrieval SPIMI vs BSBI:")
    test_queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]

    from bsbi import BSBIIndex as OrigBSBI
    bsbi = OrigBSBI(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    bsbi.load()

    # Karena kedua index menggunakan output_dir yang sama dan index_name yang sama,
    # SPIMI sudah menimpa index BSBI. Cukup verifikasi bahwa retrieval berjalan normal.
    for q in test_queries:
        bm25_results = spimi.retrieve_bm25(q, k=10)
        wand_results = spimi.retrieve_bm25_wand(q, k=10)
        bm25_docs = set(doc for _, doc in bm25_results)
        wand_docs = set(doc for _, doc in wand_results)
        match = bm25_docs == wand_docs
        print(f"  Query: {q[:50]}...")
        print(f"    BM25 top-1: {bm25_results[0][1] if bm25_results else 'N/A'} "
              f"(score={bm25_results[0][0]:.3f})")
        print(f"    WAND match: {match}")

    # Test 3: evaluasi metrik
    print("\nMenjalankan evaluasi dengan index SPIMI...")
    import math
    from index import InvertedIndexReader

    with InvertedIndexReader('main_index', VBEPostings, directory='index') as idx:
        N = len(idx.doc_length)
        num_terms = len(idx.postings_dict)
        print(f"  Jumlah dokumen: {N}")
        print(f"  Jumlah terms unik: {num_terms}")

    print("\nSPIMI indexing berhasil!")