from preprocessing import preprocess


def levenshtein_distance(s1, s2):
    """
    Menghitung Levenshtein distance (edit distance) antara dua string
    menggunakan dynamic programming.

    Levenshtein distance adalah jumlah minimum operasi edit (insert,
    delete, replace) yang diperlukan untuk mengubah s1 menjadi s2.

    Parameters
    ----------
    s1 : str
        String pertama.
    s2 : str
        String kedua.

    Returns
    -------
    int
        Levenshtein distance antara s1 dan s2.
    """
    m, n = len(s1), len(s2)
    # Optimasi: gunakan dua baris saja (bukan matriks penuh)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j],      # delete
                                  curr[j - 1],  # insert
                                  prev[j - 1])  # replace
        prev, curr = curr, prev

    return prev[n]


def _generate_candidates_edit1(word):
    """
    Menghasilkan semua string yang berjarak edit distance 1 dari word.

    Operasi yang dilakukan:
    - Deletes: hapus satu karakter
    - Transposes: tukar dua karakter bersebelahan
    - Replaces: ganti satu karakter dengan karakter lain
    - Inserts: sisipkan satu karakter

    Parameters
    ----------
    word : str
        Kata yang akan di-generate kandidatnya.

    Returns
    -------
    set[str]
        Himpunan semua string berjarak edit distance 1.
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)


class SpellCorrector:
    """
    Spelling corrector berbasis Levenshtein distance untuk query search engine.

    Corrector ini bekerja pada level stemmed terms (setelah preprocessing),
    sehingga vocabulary-nya adalah kumpulan stem yang ada di index.
    Strategi koreksi:
    1. Jika kata sudah ada di vocabulary -> kembalikan apa adanya
    2. Cari kandidat edit distance 1 yang ada di vocabulary
    3. Jika tidak ada, cari kandidat edit distance 2
    4. Pilih kandidat dengan document frequency (DF) tertinggi

    Attributes
    ----------
    vocabulary : set[str]
        Himpunan semua term (stem) yang ada di index.
    term_to_df : dict[str, int]
        Mapping dari term ke document frequency-nya.
    """

    def __init__(self, term_id_map, postings_dict):
        """
        Parameters
        ----------
        term_id_map : IdMap
            Mapping term <-> termID dari index.
        postings_dict : dict
            Dictionary dari inverted index: termID -> (start_pos, df, ...).
        """
        self.vocabulary = set()
        self.term_to_df = {}
        for term_id in range(len(term_id_map)):
            term = term_id_map[term_id]
            df = postings_dict[term_id][1] if term_id in postings_dict else 0
            self.vocabulary.add(term)
            self.term_to_df[term] = df

    def _best_candidate(self, candidates):
        """
        Memilih kandidat terbaik dari himpunan kandidat berdasarkan DF tertinggi.

        Parameters
        ----------
        candidates : set[str]
            Himpunan kandidat yang ada di vocabulary.

        Returns
        -------
        str or None
            Kandidat dengan DF tertinggi, atau None jika tidak ada.
        """
        if not candidates:
            return None
        return max(candidates, key=lambda w: self.term_to_df.get(w, 0))

    def correct_word(self, word):
        """
        Mengoreksi sebuah kata (sudah di-stem) menggunakan Levenshtein distance.

        Parameters
        ----------
        word : str
            Kata (stem) yang akan dikoreksi.

        Returns
        -------
        str
            Kata yang sudah dikoreksi (atau kata asli jika sudah benar).
        """
        if word in self.vocabulary:
            return word

        # Edit distance 1
        candidates_1 = _generate_candidates_edit1(word) & self.vocabulary
        best = self._best_candidate(candidates_1)
        if best:
            return best

        # Edit distance 2
        candidates_2 = set()
        for w in _generate_candidates_edit1(word):
            candidates_2 |= _generate_candidates_edit1(w) & self.vocabulary
        best = self._best_candidate(candidates_2)
        if best:
            return best

        return word

    def correct_query(self, query):
        """
        Mengoreksi seluruh query. Setiap kata di-preprocess (lowercase,
        tokenisasi, stopword removal, stemming) lalu dikoreksi satu per satu.

        Parameters
        ----------
        query : str
            Query mentah dari pengguna.

        Returns
        -------
        tuple[str, bool]
            Tuple berisi (query yang sudah dikoreksi, apakah ada perubahan).
            Query yang dikembalikan berisi stemmed terms yang dipisahkan spasi.
        """
        stemmed_tokens = preprocess(query)
        corrected = []
        changed = False
        for token in stemmed_tokens:
            corrected_token = self.correct_word(token)
            if corrected_token != token:
                changed = True
            corrected.append(corrected_token)
        return " ".join(corrected), changed


if __name__ == "__main__":
    from bsbi import BSBIIndex
    from compression import VBEPostings
    from index import InvertedIndexReader

    # Bangun SpellCorrector dari index
    bsbi = BSBIIndex(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    bsbi.load()

    with InvertedIndexReader('main_index', VBEPostings, directory='index') as idx:
        corrector = SpellCorrector(bsbi.term_id_map, idx.postings_dict)

    print(f"Vocabulary size: {len(corrector.vocabulary)} terms\n")

    # Test 1: Levenshtein distance
    print("=== Levenshtein Distance ===")
    test_pairs = [
        ("kitten", "sitting", 3),
        ("saturday", "sunday", 3),
        ("blood", "blood", 0),
        ("blood", "flood", 1),
        ("cell", "cel", 1),
    ]
    for s1, s2, expected in test_pairs:
        dist = levenshtein_distance(s1, s2)
        status = "OK" if dist == expected else "FAIL"
        print(f"  d('{s1}', '{s2}') = {dist} (expected {expected}) [{status}]")

    # Test 2: koreksi kata tunggal
    print("\n=== Word Correction ===")
    test_words = [
        ("blood", "sudah benar"),
        ("blod", "typo: missing 'o'"),
        ("cll", "typo: missing 'e'"),
        ("proteiin", "typo: extra 'i'"),
        ("disaese", "typo: swapped letters"),
    ]
    for word, desc in test_words:
        stemmed = preprocess(word)
        if stemmed:
            corrected = corrector.correct_word(stemmed[0])
            print(f"  '{word}' -> stem '{stemmed[0]}' -> corrected '{corrected}' ({desc})")
        else:
            print(f"  '{word}' -> dihapus oleh stopword filter ({desc})")

    # Test 3: koreksi query
    print("\n=== Query Correction ===")
    test_queries = [
        "blod presure in pregancy",
        "lipd metabolism toxmia",
        "alkylated radioactive iodoacetate",
        "psychodrama disturbed chlidren",
    ]
    for q in test_queries:
        corrected, changed = corrector.correct_query(q)
        original_stems = " ".join(preprocess(q))
        if changed:
            print(f"  '{q}'")
            print(f"    stems:     {original_stems}")
            print(f"    corrected: {corrected}")
        else:
            print(f"  '{q}' -> tidak ada koreksi (sudah benar)")