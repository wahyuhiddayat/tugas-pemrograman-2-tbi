import re

# Daftar stopwords bahasa Inggris (198 kata) dari nltk.corpus.stopwords.words('english').
# Didefinisikan langsung di sini karena file compression.py berkonflik
# dengan modul standar 'compression' di Python >= 3.14, sehingga import nltk gagal.
STOPWORDS = frozenset([
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
    "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have",
    "haven", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn", "isn't", "it", "it'd",
    "it'll", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn",
    "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn",
    "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "re", "s", "same", "shan", "shan't", "she", "she'd", "she'll", "she's",
    "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t",
    "than", "that", "that'll", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "ve", "very", "was", "wasn", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn",
    "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
])


class PorterStemmer:
    """
    Implementasi algoritma Porter Stemmer dari scratch (tanpa library
    stemmer eksternal).

    Algoritma ini terdiri dari 5 langkah utama yang secara berurutan
    menghapus suffix-suffix morfologis dari kata bahasa Inggris untuk
    mendapatkan stem (akar kata). Setiap langkah memiliki aturan-aturan
    yang diterapkan berdasarkan "measure" (m) dari stem yang dihasilkan.

    Measure m adalah jumlah pasangan (konsonan-grup, vokal-grup) dalam
    stem. Pola: [C](VC){m}[V].
    Contoh: "tree" -> m=0, "trouble" -> m=1, "troubles" -> m=2.
    """

    _VOWELS = frozenset("aeiou")

    def _is_consonant(self, word, i):
        """Cek apakah karakter ke-i dalam word adalah konsonan.
        Huruf 'y' dianggap konsonan jika didahului vokal, dan sebaliknya."""
        if word[i] in self._VOWELS:
            return False
        if word[i] == 'y':
            if i == 0:
                return True
            return not self._is_consonant(word, i - 1)
        return True

    def _measure(self, stem):
        """
        Menghitung measure (m) dari sebuah stem.
        m adalah jumlah pasangan VC (vokal-grup diikuti konsonan-grup).
        Pola: [C](VC){m}[V]
        """
        n = len(stem)
        i = 0
        while i < n and self._is_consonant(stem, i):
            i += 1
        m = 0
        while i < n:
            while i < n and not self._is_consonant(stem, i):
                i += 1
            if i >= n:
                break
            while i < n and self._is_consonant(stem, i):
                i += 1
            m += 1
        return m

    def _has_vowel(self, stem):
        """Cek apakah stem mengandung setidaknya satu vokal."""
        return any(not self._is_consonant(stem, i) for i in range(len(stem)))

    def _ends_double_consonant(self, word):
        """Cek apakah word diakhiri dengan konsonan ganda yang sama (misal 'll', 'ss')."""
        if len(word) < 2:
            return False
        return (word[-1] == word[-2] and self._is_consonant(word, len(word) - 1))

    def _ends_cvc(self, word):
        """
        Cek apakah word diakhiri dengan pola konsonan-vokal-konsonan,
        di mana konsonan terakhir bukan w, x, atau y.
        Digunakan sebagai kondisi tambahan di beberapa aturan.
        """
        if len(word) < 3:
            return False
        return (self._is_consonant(word, len(word) - 1) and
                not self._is_consonant(word, len(word) - 2) and
                self._is_consonant(word, len(word) - 3) and
                word[-1] not in ('w', 'x', 'y'))

    def _strip_plurals(self, word):
        """
        Langkah 1a dari algoritma Porter.
        Menghapus suffix plural (-sses, -ies, -s) dari kata.
        Contoh: caresses -> caress, ponies -> poni, cats -> cat.
        """
        if word.endswith('sses'):
            return word[:-2]
        if word.endswith('ies'):
            return word[:-2]
        if word.endswith('ss'):
            return word
        if word.endswith('s'):
            return word[:-1]
        return word

    def _strip_past_participle(self, word):
        """
        Langkah 1b dari algoritma Porter.
        Menghapus suffix past tense dan gerund (-eed, -ed, -ing).
        Setelah penghapusan -ed/-ing, ada aturan tambahan untuk
        memperbaiki stem (misal: hopping -> hop, not hopp).
        """
        if word.endswith('eed'):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return word[:-1]
            return word
        for suffix in ('ed', 'ing'):
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._has_vowel(stem):
                    word = stem
                    if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                        return word + 'e'
                    if (self._ends_double_consonant(word) and
                            word[-1] not in ('l', 's', 'z')):
                        return word[:-1]
                    if self._measure(word) == 1 and self._ends_cvc(word):
                        return word + 'e'
                    return word
                return word + suffix
        return word

    def _replace_y_suffix(self, word):
        """
        Langkah 1c dari algoritma Porter.
        Mengganti trailing 'y' dengan 'i' jika stem mengandung vokal.
        Contoh: happy -> happi, sky -> sky (tidak berubah, stem 'sk' tanpa vokal).
        """
        if word.endswith('y'):
            stem = word[:-1]
            if self._has_vowel(stem):
                return stem + 'i'
        return word

    def _map_double_suffixes(self, word):
        """
        Langkah 2 dari algoritma Porter.
        Memetakan suffix derivasional ganda ke bentuk yang lebih sederhana.
        Contoh: relational -> relate, conditional -> condition,
                digitizer -> digitize, vietnamization -> vietnam.
        Hanya diterapkan jika measure(stem) > 0.
        """
        pairs = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
            ('anci', 'ance'), ('izer', 'ize'), ('abli', 'able'),
            ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'),
            ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
            ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'),
            ('fulness', 'ful'), ('ousness', 'ous'), ('aliti', 'al'),
            ('iviti', 'ive'), ('biliti', 'ble'),
        ]
        for old, new in pairs:
            if word.endswith(old):
                stem = word[:-len(old)]
                if self._measure(stem) > 0:
                    return stem + new
                return word
        return word

    def _map_single_suffixes(self, word):
        """
        Langkah 3 dari algoritma Porter.
        Memetakan suffix derivasional tunggal ke bentuk lebih sederhana.
        Contoh: triplicate -> triplic, formalize -> formal,
                hopeful -> hope, goodness -> good.
        Hanya diterapkan jika measure(stem) > 0.
        """
        pairs = [
            ('icate', 'ic'), ('ative', ''), ('alize', 'al'),
            ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''), ('ness', ''),
        ]
        for old, new in pairs:
            if word.endswith(old):
                stem = word[:-len(old)]
                if self._measure(stem) > 0:
                    return stem + new
                return word
        return word

    def _remove_derivational_suffixes(self, word):
        """
        Langkah 4 dari algoritma Porter.
        Menghapus berbagai suffix derivasional (-al, -ance, -ence, -er, dst.)
        hanya jika measure(stem) > 1 (stem harus cukup panjang).
        Contoh: revival -> reviv, allowance -> allow, inference -> infer.
        Khusus suffix -ion: stem harus diakhiri 's' atau 't'.
        """
        suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant',
            'ement', 'ment', 'ent', 'ion', 'ou', 'ism', 'ate', 'iti',
            'ous', 'ive', 'ize',
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if suffix == 'ion':
                    if self._measure(stem) > 1 and len(stem) > 0 and stem[-1] in ('s', 't'):
                        return stem
                else:
                    if self._measure(stem) > 1:
                        return stem
                return word
        return word

    def _clean_trailing_e(self, word):
        """
        Langkah 5a dari algoritma Porter.
        Menghapus trailing 'e' jika measure > 1, atau jika measure == 1
        dan stem tidak diakhiri pola CVC.
        Contoh: probate -> probat, rate -> rate (tidak berubah).
        """
        if word.endswith('e'):
            stem = word[:-1]
            if self._measure(stem) > 1:
                return stem
            if self._measure(stem) == 1 and not self._ends_cvc(stem):
                return stem
        return word

    def _clean_double_l(self, word):
        """
        Langkah 5b dari algoritma Porter.
        Menghapus salah satu 'l' dari akhiran -ll jika measure > 1.
        Contoh: controll -> control, roll -> roll (tidak berubah, m=1).
        """
        if (self._measure(word) > 1 and
                self._ends_double_consonant(word) and word[-1] == 'l'):
            return word[:-1]
        return word

    def stem(self, word):
        """
        Mengembalikan stem dari sebuah kata bahasa Inggris menggunakan
        algoritma Porter (5 langkah berurutan).

        Parameters
        ----------
        word : str
            Kata yang akan di-stem (harus sudah lowercase).

        Returns
        -------
        str
            Stem dari kata tersebut.
        """
        if len(word) <= 2:
            return word

        # Langkah 1: infleksional (plural, past tense, gerund, trailing y)
        word = self._strip_plurals(word)
        word = self._strip_past_participle(word)
        word = self._replace_y_suffix(word)

        # Langkah 2-3: suffix derivasional
        word = self._map_double_suffixes(word)
        word = self._map_single_suffixes(word)

        # Langkah 4: hapus suffix derivasional (stem harus cukup panjang)
        word = self._remove_derivational_suffixes(word)

        # Langkah 5: pembersihan akhir
        word = self._clean_trailing_e(word)
        word = self._clean_double_l(word)

        return word


_stemmer = PorterStemmer()


def preprocess(text):
    """
    Pipeline preprocessing lengkap untuk teks bahasa Inggris:
    1. Lowercase
    2. Tokenisasi dengan regex (hanya karakter alfabet)
    3. Buang stopwords (dari NLTK, 198 kata)
    4. Stemming (Porter Stemmer)

    Parameters
    ----------
    text : str
        Teks mentah yang akan diproses.

    Returns
    -------
    list[str]
        List of stemmed tokens (tanpa stopwords).
    """
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    return [_stemmer.stem(t) for t in tokens if t not in STOPWORDS]


if __name__ == "__main__":
    stemmer = PorterStemmer()

    # Test case dari paper Porter
    test_cases = [
        ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"),
        ("caress", "caress"), ("cats", "cat"),
        ("feed", "feed"), ("agreed", "agre"), ("disabled", "disabl"),
        ("matting", "mat"), ("mating", "mate"), ("meeting", "meet"),
        ("milling", "mill"), ("messing", "mess"), ("meetings", "meet"),
        ("happi", "happi"), ("sky", "sky"),
        ("relational", "relat"), ("conditional", "condit"),
        ("rational", "ration"), ("valenci", "valenc"),
        ("hesitanci", "hesit"), ("digitizer", "digit"),
        ("conformabli", "conform"), ("radicalli", "radic"),
        ("differentli", "differ"), ("vileli", "vile"),
        ("analogousli", "analog"), ("vietnamization", "vietnam"),
        ("predication", "predic"), ("operator", "oper"),
        ("feudalism", "feudal"), ("decisiveness", "decis"),
        ("hopefulness", "hope"), ("callousness", "callous"),
        ("formaliti", "formal"), ("sensitiviti", "sensit"),
        ("sensibiliti", "sensibl"),
        ("triplicate", "triplic"), ("formative", "form"),
        ("formalize", "formal"), ("electriciti", "electr"),
        ("electrical", "electr"), ("hopeful", "hope"),
        ("goodness", "good"),
        ("revival", "reviv"), ("allowance", "allow"),
        ("inference", "infer"), ("airliner", "airlin"),
        ("gyroscopic", "gyroscop"), ("adjustable", "adjust"),
        ("defensible", "defens"), ("irritant", "irrit"),
        ("replacement", "replac"), ("adjustment", "adjust"),
        ("dependent", "depend"), ("adoption", "adopt"),
        ("homologou", "homolog"), ("communism", "commun"),
        ("activate", "activ"), ("angulariti", "angular"),
        ("homologous", "homolog"), ("effective", "effect"),
        ("bowdlerize", "bowdler"),
        ("probate", "probat"), ("rate", "rate"),
        ("cease", "ceas"),
        ("controll", "control"), ("roll", "roll"),
    ]

    passed = 0
    failed = 0
    for word, expected in test_cases:
        result = stemmer.stem(word)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: stem({word!r}) = {result!r}, expected {expected!r}")

    print(f"\nPorter Stemmer: {passed}/{passed + failed} test cases passed")

    # Test preprocess pipeline
    text = "The quick brown foxes are jumping over the lazy dogs"
    tokens = preprocess(text)
    print(f"\npreprocess({text!r})")
    print(f"  -> {tokens}")
    assert "the" not in tokens, "stopword 'the' seharusnya sudah dibuang"
    assert "are" not in tokens, "stopword 'are' seharusnya sudah dibuang"
    assert "over" not in tokens, "stopword 'over' seharusnya sudah dibuang"
    print("  stopword removal: OK")
    print(f"  jumlah stopwords (NLTK): {len(STOPWORDS)}")
    print("  pipeline: OK")