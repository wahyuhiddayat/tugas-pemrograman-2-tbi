import pickle
import heapq


class TrieNode:
    """
    Node pada struktur data Trie.

    Attributes
    ----------
    children : dict
        Mapping dari karakter ke TrieNode anak.
    is_terminal : bool
        True jika node ini merepresentasikan akhir dari sebuah term.
    term : str
        Term lengkap yang tersimpan di node terminal (None jika bukan terminal).
    term_id : int
        Term ID yang berkorespondensi dengan term ini di IdMap (None jika bukan terminal).
    df : int
        Document frequency dari term ini, yaitu jumlah dokumen yang mengandung
        term tersebut. Digunakan untuk perankingan pada autocomplete.
    """

    def __init__(self):
        self.children = {}
        self.is_terminal = False
        self.term = None
        self.term_id = None
        self.df = 0


class Trie:
    """
    Implementasi struktur data Trie untuk dictionary pada search engine.

    Trie menyimpan seluruh term yang ada di index dan memungkinkan operasi
    yang tidak bisa dilakukan oleh hash-based dictionary (seperti IdMap):
    - Prefix search: mencari semua term yang diawali prefix tertentu
    - Autocomplete: mengembalikan top-K term berdasarkan document frequency

    Trie ini melengkapi (bukan menggantikan) IdMap. IdMap tetap digunakan
    untuk mapping term <-> termID yang cepat (O(1)), sedangkan Trie digunakan
    untuk fitur prefix-based query.

    Attributes
    ----------
    root : TrieNode
        Root node dari Trie (tidak merepresentasikan karakter apapun).
    size : int
        Jumlah term yang tersimpan di Trie.
    """

    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, term, term_id, df=0):
        """
        Memasukkan sebuah term ke dalam Trie.

        Parameters
        ----------
        term : str
            Term yang akan dimasukkan.
        term_id : int
            Term ID yang berkorespondensi dengan term ini di IdMap.
        df : int
            Document frequency dari term ini.
        """
        node = self.root
        for char in term:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_terminal = True
        node.term = term
        node.term_id = term_id
        node.df = df
        self.size += 1

    def search(self, term):
        """
        Mencari sebuah term secara eksak di Trie.

        Parameters
        ----------
        term : str
            Term yang dicari.

        Returns
        -------
        TrieNode or None
            Node terminal jika term ditemukan, None jika tidak.
        """
        node = self.root
        for char in term:
            if char not in node.children:
                return None
            node = node.children[char]
        return node if node.is_terminal else None

    def _find_prefix_node(self, prefix):
        """
        Mencari node yang berkorespondensi dengan akhir dari sebuah prefix.

        Parameters
        ----------
        prefix : str
            Prefix yang dicari.

        Returns
        -------
        TrieNode or None
            Node di akhir prefix, atau None jika prefix tidak ada di Trie.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def prefix_search(self, prefix):
        """
        Mencari semua term yang diawali dengan prefix tertentu.

        Parameters
        ----------
        prefix : str
            Prefix yang dicari.

        Returns
        -------
        list[tuple[str, int, int]]
            List of (term, term_id, df) untuk semua term yang cocok,
            diurutkan secara alfabet.
        """
        node = self._find_prefix_node(prefix)
        if node is None:
            return []

        results = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.is_terminal:
                results.append((current.term, current.term_id, current.df))
            for char in sorted(current.children.keys(), reverse=True):
                stack.append(current.children[char])
        return sorted(results, key=lambda x: x[0])

    def autocomplete(self, prefix, top_k=10):
        """
        Mengembalikan top-K term yang diawali dengan prefix tertentu,
        diurutkan berdasarkan document frequency (DF) dari tertinggi ke terendah.

        Menggunakan min-heap untuk efisiensi pada Trie yang besar: hanya
        mempertahankan K term dengan DF tertinggi tanpa perlu menyimpan
        semua hasil prefix search.

        Parameters
        ----------
        prefix : str
            Prefix yang dicari.
        top_k : int
            Jumlah term yang dikembalikan (default 10).

        Returns
        -------
        list[tuple[str, int]]
            List of (term, df) diurutkan dari DF tertinggi ke terendah.
        """
        node = self._find_prefix_node(prefix)
        if node is None:
            return []

        heap = []  # min-heap: (df, term)
        stack = [node]
        while stack:
            current = stack.pop()
            if current.is_terminal:
                if len(heap) < top_k:
                    heapq.heappush(heap, (current.df, current.term))
                elif current.df > heap[0][0]:
                    heapq.heapreplace(heap, (current.df, current.term))
            for child in current.children.values():
                stack.append(child)

        results = []
        while heap:
            df, term = heapq.heappop(heap)
            results.append((term, df))
        results.reverse()
        return results

    def build_from_index(self, term_id_map, postings_dict):
        """
        Membangun Trie dari index yang sudah ada.

        Mengiterasi seluruh term di term_id_map dan memasukkannya ke Trie
        beserta document frequency-nya (dari postings_dict).

        Parameters
        ----------
        term_id_map : IdMap
            Mapping term <-> termID.
        postings_dict : dict
            Dictionary dari inverted index: termID -> (start_pos, df, ...).
        """
        for term_id in range(len(term_id_map)):
            term = term_id_map[term_id]
            df = postings_dict[term_id][1] if term_id in postings_dict else 0
            self.insert(term, term_id, df)

    def save(self, filepath):
        """Menyimpan Trie ke file menggunakan pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Memuat Trie dari file pickle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    import os
    from bsbi import BSBIIndex
    from compression import VBEPostings
    from index import InvertedIndexReader

    # Bangun Trie dari index yang sudah ada
    bsbi = BSBIIndex(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    bsbi.load()

    with InvertedIndexReader('main_index', VBEPostings, directory='index') as idx:
        trie = Trie()
        trie.build_from_index(bsbi.term_id_map, idx.postings_dict)

    print(f"Trie berhasil dibangun: {trie.size} terms\n")

    # Test 1: exact search
    print("=== Exact Search ===")
    test_terms = ["blood", "cell", "xyz_not_exist", "protein"]
    for term in test_terms:
        from preprocessing import preprocess
        stemmed = preprocess(term)
        if stemmed:
            node = trie.search(stemmed[0])
            if node:
                print(f"  '{term}' -> stem '{stemmed[0]}': ditemukan (DF={node.df})")
            else:
                print(f"  '{term}' -> stem '{stemmed[0]}': tidak ditemukan")
        else:
            print(f"  '{term}': dihapus oleh stopword filter")

    # Test 2: prefix search
    print("\n=== Prefix Search ===")
    test_prefixes = ["blo", "cel", "prot", "xyz"]
    for prefix in test_prefixes:
        results = trie.prefix_search(prefix)
        print(f"  prefix '{prefix}': {len(results)} terms ditemukan", end="")
        if results:
            shown = results[:5]
            terms_str = ", ".join(f"{t}(DF={d})" for t, _, d in shown)
            if len(results) > 5:
                terms_str += f", ... (+{len(results)-5} lagi)"
            print(f" -> [{terms_str}]")
        else:
            print()

    # Test 3: autocomplete (top-5 by DF)
    print("\n=== Autocomplete (top-5 by DF) ===")
    test_prefixes = ["blo", "cel", "prot", "dis"]
    for prefix in test_prefixes:
        results = trie.autocomplete(prefix, top_k=5)
        if results:
            terms_str = ", ".join(f"{t}(DF={d})" for t, d in results)
            print(f"  '{prefix}' -> [{terms_str}]")
        else:
            print(f"  '{prefix}' -> tidak ada hasil")

    # Test 4: save dan load
    trie_path = os.path.join('index', 'trie.pkl')
    trie.save(trie_path)
    trie_loaded = Trie.load(trie_path)
    assert trie_loaded.size == trie.size, "Jumlah term tidak cocok setelah load"
    node1 = trie.search("blood")
    if node1:
        from preprocessing import preprocess
        stemmed = preprocess("blood")[0]
        node2 = trie_loaded.search(stemmed)
        assert node2 is not None and node2.df == node1.df, "DF tidak cocok setelah load"
    print(f"\nSave/load berhasil: {trie_path} ({trie_loaded.size} terms)")