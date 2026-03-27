import re
import os

from preprocessing import preprocess, PorterStemmer

_stemmer = PorterStemmer()

def _stem_word(word):
    """Stem satu kata menggunakan Porter Stemmer."""
    return _stemmer.stem(word.lower())


class SnippetGenerator:
    """
    Pembuat snippet dokumen berbasis KWIC (Key Word In Context).

    Untuk setiap dokumen hasil retrieval, generator ini mencari jendela teks
    dengan kepadatan query term tertinggi, lalu mengembalikan potongan teks
    tersebut sebagai snippet yang ditampilkan kepada pengguna.

    Algoritma:
    1. Tokenisasi dokumen menjadi daftar kata (original case).
    2. Stem setiap kata; tandai posisi yang cocok dengan stem query.
    3. Gunakan sliding window berukuran `window_size` kata untuk menemukan
       jendela dengan hit terbanyak.
    4. Kembalikan teks jendela tersebut dengan marker elipsis jika diperlukan.
    """

    def __init__(self, window_size=50):
        """
        Inisialisasi SnippetGenerator.

        Parameters
        ----------
        window_size : int
            Jumlah kata dalam satu jendela snippet.
        """
        self.window_size = window_size

    def _tokenize(self, text):
        """
        Tokenisasi teks menjadi daftar (token, original_word) pairs.

        Memisahkan berdasarkan spasi dan tanda baca, mempertahankan kata
        asli untuk rekonstruksi snippet.
        """
        words = re.findall(r"[A-Za-z0-9']+", text)
        return words

    def _find_hit_positions(self, words, query_stems):
        """
        Temukan posisi kata-kata yang stem-nya cocok dengan query stems.

        Parameters
        ----------
        words : list[str]
            Daftar kata dari dokumen.
        query_stems : set[str]
            Himpunan stem dari query terms.

        Returns
        -------
        list[bool]
            Boolean mask; True berarti kata di posisi tersebut adalah hit.
        """
        hits = []
        for word in words:
            stemmed = _stem_word(word)
            hits.append(stemmed in query_stems)
        return hits

    def _best_window(self, hits):
        """
        Temukan indeks awal jendela dengan jumlah hit terbanyak menggunakan
        sliding window O(N).

        Parameters
        ----------
        hits : list[bool]
            Boolean mask dari posisi hit.

        Returns
        -------
        int
            Indeks awal jendela terbaik.
        """
        n = len(hits)
        w = min(self.window_size, n)

        # Hitung jumlah hit pada jendela pertama
        current = sum(hits[:w])
        best_count = current
        best_start = 0

        for i in range(1, n - w + 1):
            current += hits[i + w - 1] - hits[i - 1]
            if current > best_count:
                best_count = current
                best_start = i

        return best_start

    def generate(self, doc_path, query):
        """
        Hasilkan snippet KWIC untuk dokumen dan query yang diberikan.

        Parameters
        ----------
        doc_path : str
            Path ke file dokumen teks.
        query : str
            Query string dari pengguna.

        Returns
        -------
        str
            Snippet teks dengan elipsis sebagai penanda konteks.
            Jika dokumen tidak bisa dibaca, kembalikan string kosong.
        """
        try:
            with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            return ""

        words = self._tokenize(text)
        if not words:
            return ""

        query_stems = set(preprocess(query))
        if not query_stems:
            # Tidak ada query term valid, kembalikan awal dokumen saja
            snippet_words = words[:self.window_size]
            return " ".join(snippet_words) + (" ..." if len(words) > self.window_size else "")

        hits = self._find_hit_positions(words, query_stems)
        start = self._best_window(hits)
        end = min(start + self.window_size, len(words))

        snippet_words = words[start:end]
        snippet = " ".join(snippet_words)

        prefix = "... " if start > 0 else ""
        suffix = " ..." if end < len(words) else ""

        return prefix + snippet + suffix

    def generate_batch(self, doc_paths, query):
        """
        Hasilkan snippet untuk beberapa dokumen sekaligus.

        Parameters
        ----------
        doc_paths : list[str]
            Daftar path dokumen.
        query : str
            Query string.

        Returns
        -------
        list[str]
            Daftar snippet, satu per dokumen.
        """
        return [self.generate(path, query) for path in doc_paths]


if __name__ == "__main__":

    collection_root = os.path.join(
        os.path.dirname(__file__), "collection"
    )

    gen = SnippetGenerator(window_size=30)

    test_cases = [
        ("collection/1/1.txt", "glucose fetal plasma"),
        ("collection/1/5.txt", "blood pressure pregnancy"),
        ("collection/2/100.txt", "protein synthesis"),
    ]

    for rel_path, query in test_cases:
        doc_path = os.path.join(os.path.dirname(__file__), rel_path)
        snippet = gen.generate(doc_path, query)
        print(f"Query : {query!r}")
        print(f"Doc   : {rel_path}")
        print(f"Snippet: {snippet}")
        print()