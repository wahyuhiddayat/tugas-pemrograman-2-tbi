import os
import time

from bsbi import BSBIIndex
from compression import VBEPostings
from index import InvertedIndexReader
from preprocessing import preprocess
from trie import Trie
from spell_correction import SpellCorrector
from query_expansion import RocchioQueryExpansion
from lsi import LSIRetriever
from snippets import SnippetGenerator


MODES = ['tfidf', 'bm25', 'wand', 'lsi', 'prf']

HELP_TEXT = """
Perintah yang tersedia:
  :mode <nama>      Ganti mode retrieval. Pilihan: tfidf, bm25, wand, lsi, prf
  :k <angka>        Ganti jumlah hasil yang ditampilkan (default: 10)
  :ac <prefix>      Tampilkan autocomplete dari prefix (contoh: :ac dis)
  :spell on/off     Aktifkan atau nonaktifkan koreksi ejaan otomatis
  :help             Tampilkan bantuan ini
  :quit / :q        Keluar dari program

Mode retrieval:
  tfidf  - TF-IDF dengan scoring TaaT
  bm25   - BM25 (Okapi BM25) dengan scoring TaaT
  wand   - BM25 + WAND top-K pruning
  lsi    - Latent Semantic Indexing (SVD, k=100)
  prf    - BM25 + Pseudo-Relevance Feedback (Rocchio)

Ketik query langsung untuk mencari. Koreksi ejaan otomatis diterapkan jika
term query tidak ditemukan di index (gunakan :spell off untuk menonaktifkan).
"""


class InteractiveSearch:
    """
    REPL interaktif untuk sistem information retrieval.

    Mengintegrasikan semua komponen yang telah diimplementasikan:
    - Retrieval: TF-IDF, BM25, BM25+WAND, LSI, BM25+PRF
    - Spelling correction berbasis Levenshtein distance
    - Autocomplete berbasis Trie dengan ranking DF
    - Snippet generation berbasis KWIC

    Attributes
    ----------
    bsbi : BSBIIndex
        Instance BSBIIndex yang sudah di-load.
    trie : Trie
        Trie dictionary untuk autocomplete.
    spell_corrector : SpellCorrector
        Spell corrector berbasis edit distance.
    spell_enabled : bool
        Jika True, koreksi ejaan diterapkan otomatis pada setiap query.
    prf : RocchioQueryExpansion
        Query expansion dengan Rocchio PRF.
    lsi : LSIRetriever or None
        LSI retriever (None jika model tidak tersedia).
    snippet_gen : SnippetGenerator
        Generator snippet KWIC.
    mode : str
        Mode retrieval aktif saat ini.
    top_k : int
        Jumlah hasil yang ditampilkan.
    """

    def __init__(self, data_dir='collection', index_dir='index'):
        """
        Inisialisasi semua komponen dan muat index dari disk.

        Parameters
        ----------
        data_dir : str
            Direktori koleksi dokumen.
        index_dir : str
            Direktori tempat index tersimpan.
        """
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.mode = 'bm25'
        self.top_k = 10
        self.spell_enabled = True

        print("Memuat komponen search engine...")
        self._load_index()
        self._load_trie()
        self._load_spell_corrector()
        self._load_prf()
        self._load_lsi()
        self.snippet_gen = SnippetGenerator(window_size=35)
        print("Siap.\n")

    def _load_index(self):
        """Memuat BSBI index dari disk."""
        self.bsbi = BSBIIndex(
            data_dir=self.data_dir,
            postings_encoding=VBEPostings,
            output_dir=self.index_dir,
        )
        self.bsbi.load()
        print(f"  [OK] Index dimuat: {len(self.bsbi.doc_id_map)} dokumen, "
              f"{len(self.bsbi.term_id_map)} terms")

    def _build_trie(self, trie_path):
        """Membangun Trie dari index dan simpan ke disk."""
        with InvertedIndexReader('main_index', VBEPostings, directory=self.index_dir) as idx:
            self.trie = Trie()
            self.trie.build_from_index(self.bsbi.term_id_map, idx.postings_dict)
        self.trie.save(trie_path)
        print(f"  [OK] Trie dibangun dan disimpan: {self.trie.size} terms")

    def _load_trie(self):
        """Memuat atau membangun Trie untuk autocomplete."""
        trie_path = os.path.join(self.index_dir, 'trie.pkl')
        if os.path.exists(trie_path):
            try:
                self.trie = Trie.load(trie_path)
                print(f"  [OK] Trie dimuat: {self.trie.size} terms")
            except (AttributeError, Exception):
                # File lama disimpan sebagai __main__.Trie, rebuild
                print("  Trie lama tidak kompatibel, membangun ulang...")
                self._build_trie(trie_path)
        else:
            print("  Membangun Trie dari index...")
            self._build_trie(trie_path)

    def _load_spell_corrector(self):
        """Memuat SpellCorrector dari postings_dict index."""
        with InvertedIndexReader('main_index', VBEPostings, directory=self.index_dir) as idx:
            self.spell_corrector = SpellCorrector(
                self.bsbi.term_id_map, idx.postings_dict
            )
        print(f"  [OK] Spell corrector siap "
              f"({len(self.spell_corrector.vocabulary)} terms)")

    def _load_prf(self):
        """Memuat Rocchio PRF."""
        self.prf = RocchioQueryExpansion(
            self.bsbi, alpha=1.0, beta=0.75,
            num_feedback_docs=5, num_expansion_terms=10
        )
        print("  [OK] PRF (Rocchio) siap")

    def _load_lsi(self):
        """Memuat model LSI dari disk jika tersedia."""
        lsi_path = os.path.join(self.index_dir, 'lsi_model.pkl')
        if os.path.exists(lsi_path):
            self.lsi = LSIRetriever(self.bsbi)
            self.lsi.load(lsi_path)
        else:
            self.lsi = None
            print("  [--] Model LSI tidak ditemukan (lewati mode lsi)")

    def _retrieve(self, query, k):
        """
        Jalankan retrieval sesuai mode aktif.

        Parameters
        ----------
        query : str
            Query yang sudah diproses (bisa raw atau sudah di-stem).
        k : int
            Jumlah top dokumen.

        Returns
        -------
        tuple[list[tuple[float, str]], dict]
            (results, info) di mana results adalah [(score, doc_path)]
            dan info adalah metadata tambahan (expansion terms, dsb.).
        """
        info = {}
        if self.mode == 'tfidf':
            results = self.bsbi.retrieve_tfidf(query, k=k)
        elif self.mode == 'bm25':
            results = self.bsbi.retrieve_bm25(query, k=k)
        elif self.mode == 'wand':
            results = self.bsbi.retrieve_bm25_wand(query, k=k)
        elif self.mode == 'lsi':
            if self.lsi is None:
                print("  [!] Model LSI tidak tersedia. Beralih ke BM25.")
                results = self.bsbi.retrieve_bm25(query, k=k)
            else:
                results = self.lsi.retrieve(query, top_k=k)
        elif self.mode == 'prf':
            results, expanded_query, expansion_terms = \
                self.prf.retrieve_with_prf(query, k=k)
            info['expanded_query'] = expanded_query
            info['expansion_terms'] = expansion_terms
        else:
            results = self.bsbi.retrieve_bm25(query, k=k)
        return results, info

    def _display_results(self, results, query, info, elapsed):
        """
        Tampilkan hasil retrieval beserta snippet KWIC.

        Parameters
        ----------
        results : list[tuple[float, str]]
            Daftar (score, doc_path) hasil retrieval.
        query : str
            Query yang digunakan untuk retrieval (untuk snippet).
        info : dict
            Metadata tambahan (expansion terms, dsb.).
        elapsed : float
            Waktu retrieval dalam detik.
        """
        mode_label = self.mode.upper()
        print(f"\n[{mode_label}] {len(results)} hasil ({elapsed:.3f}s)\n")

        if 'expansion_terms' in info and info['expansion_terms']:
            print(f"  Expansion terms: {', '.join(info['expansion_terms'])}\n")

        if not results:
            print("  Tidak ada dokumen yang cocok.\n")
            return

        for rank, (score, doc_path) in enumerate(results, start=1):
            # Normalisasi path: hapus "./" prefix untuk tampilan bersih
            display_path = doc_path.lstrip('.').lstrip('/').lstrip('\\')

            snippet = self.snippet_gen.generate(doc_path, query)
            # Potong snippet di batas kata agar tidak memotong di tengah kata
            if len(snippet) > 200:
                snippet = snippet[:200].rsplit(' ', 1)[0] + " ..."

            print(f"  {rank:>2}. [{score:.4f}] {display_path}")
            if snippet:
                print(f"      {snippet}")
            print()

    def _handle_autocomplete(self, prefix):
        """
        Tampilkan hasil autocomplete untuk prefix yang diberikan.

        Parameters
        ----------
        prefix : str
            Prefix yang akan di-autocomplete.
        """
        if not prefix:
            print("  Penggunaan: :ac <prefix>")
            return
        # Stem prefix agar sesuai dengan vocab index (yang berisi stems)
        stemmed_tokens = preprocess(prefix)
        lookup = stemmed_tokens[0] if stemmed_tokens else prefix.lower()
        results = self.trie.autocomplete(lookup, top_k=10)
        if not results:
            print(f"  Tidak ada term dengan prefix '{prefix}'.")
        else:
            print(f"  Autocomplete '{prefix}':")
            for term, df in results:
                print(f"    {term}  (DF={df})")
        print()

    def _apply_spell_correction(self, raw_query):
        """
        Terapkan spell correction ke query dan tampilkan jika ada perubahan.
        Diabaikan jika self.spell_enabled = False.

        Parameters
        ----------
        raw_query : str
            Query mentah dari pengguna.

        Returns
        -------
        str
            Query setelah koreksi ejaan (dalam bentuk stems), atau query
            yang sudah di-preprocess jika spell correction dinonaktifkan.
        """
        if not self.spell_enabled:
            return " ".join(preprocess(raw_query))
        corrected, was_changed = self.spell_corrector.correct_query(raw_query)
        if was_changed:
            print(f"  [Spell] Koreksi: \"{raw_query}\" -> \"{corrected}\"")
        return corrected

    def run(self):
        """
        Jalankan REPL loop utama.

        Loop terus berjalan sampai pengguna mengetikkan :quit atau :q.
        Setiap input diproses sebagai perintah (diawali ':') atau query.
        """
        print("=" * 60)
        print("  Interactive Search Engine")
        print("  Ketik :help untuk bantuan, :quit untuk keluar")
        print("=" * 60)
        spell_status = "on" if self.spell_enabled else "off"
        print(f"  Mode: {self.mode.upper()}  |  Top-K: {self.top_k}  |  Spell: {spell_status}")
        print()

        while True:
            try:
                raw = input(f"[{self.mode}] Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nKeluar.")
                break

            if not raw:
                continue

            if raw.startswith(':'):
                parts = raw[1:].split(None, 1)
                cmd = parts[0].lower() if parts else ''
                arg = parts[1].strip() if len(parts) > 1 else ''

                if cmd in ('quit', 'exit', 'q'):
                    print("Sampai jumpa!")
                    break
                elif cmd == 'help':
                    print(HELP_TEXT)
                elif cmd == 'mode':
                    if arg in MODES:
                        self.mode = arg
                        print(f"  Mode diubah ke: {self.mode.upper()}\n")
                    else:
                        print(f"  Mode tidak valid. Pilihan: {', '.join(MODES)}\n")
                elif cmd == 'k':
                    if arg.isdigit() and int(arg) > 0:
                        self.top_k = int(arg)
                        print(f"  Top-K diubah ke: {self.top_k}\n")
                    else:
                        print("  Angka tidak valid. Contoh: :k 20\n")
                elif cmd in ('ac', 'autocomplete'):
                    self._handle_autocomplete(arg)
                elif cmd == 'spell':
                    if arg == 'on':
                        self.spell_enabled = True
                        print("  Koreksi ejaan: AKTIF\n")
                    elif arg == 'off':
                        self.spell_enabled = False
                        print("  Koreksi ejaan: NONAKTIF\n")
                    else:
                        status = "AKTIF" if self.spell_enabled else "NONAKTIF"
                        print(f"  Koreksi ejaan saat ini: {status}. "
                              f"Gunakan :spell on atau :spell off\n")
                else:
                    print(f"  Perintah tidak dikenal: ':{cmd}'. Ketik :help.\n")
            else:
                # Terapkan spell correction
                query = self._apply_spell_correction(raw)

                # Retrieval
                t0 = time.time()
                results, info = self._retrieve(query, self.top_k)
                elapsed = time.time() - t0

                self._display_results(results, query, info, elapsed)


if __name__ == "__main__":
    repl = InteractiveSearch(data_dir='collection', index_dir='index')
    repl.run()