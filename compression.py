import array
import struct

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class EliasGammaPostings:
    """
    Berbeda dengan VBEPostings yang beroperasi di level byte, EliasGammaPostings
    melakukan kompresi di level bit menggunakan algoritma Elias-Gamma Encoding.

    Sama seperti VBEPostings, postings list terlebih dahulu dikonversi ke
    representasi gap-based sebelum di-encode. Setiap bilangan bulat positif n
    di-encode sebagai:
      - floor(log2(n)) buah bit nol (unary code untuk magnitudo)
      - representasi biner n sepanjang floor(log2(n)) + 1 bit

    Karena docID bisa dimulai dari 0 (IdMap memberikan ID mulai dari 0),
    diterapkan offset +1 sebelum encoding dan -1 setelah decoding, sehingga
    nilai nol dapat direpresentasikan dengan aman.

    Bytestream diawali dengan unsigned integer 4-byte (big-endian) yang
    menyimpan jumlah bilangan yang di-encode. Hal ini diperlukan karena
    Elias-Gamma menghasilkan kode dengan panjang variabel yang di-pack ke
    dalam byte dengan zero-padding di akhir; tanpa jumlah tersebut kita tidak
    bisa mengetahui di mana bit valid berhenti.

    Perbandingan ukuran vs. VBE:
      - Elias-Gamma menggunakan 2*floor(log2(n))+1 bit per nilai.
      - Lebih hemat ruang dibanding VBE untuk bilangan sangat kecil (n <= 7),
        namun kurang efisien untuk bilangan besar. Cocok untuk distribusi gap
        yang sangat miring (banyak gap=1, tipikal pada postings list yang padat).

    ASUMSI: postings_list untuk sebuah term MUAT di memori!
    """

    @staticmethod
    def _encode_number_bits(n):
        """
        Mengembalikan list of bits yang meng-encode bilangan bulat n >= 1
        menggunakan Elias-Gamma.

        Untuk n >= 1:
          k = floor(log2(n))   (= n.bit_length() - 1)
          Encoding = k buah bit nol  +  representasi biner n  (k+1 bit)

        Contoh:
          n=1  -> k=0 -> [1]              (1 bit)
          n=2  -> k=1 -> [0,1,0]          (3 bit)
          n=3  -> k=1 -> [0,1,1]          (3 bit)
          n=4  -> k=2 -> [0,0,1,0,0]      (5 bit)
          n=7  -> k=2 -> [0,0,1,1,1]      (5 bit)
          n=8  -> k=3 -> [0,0,0,1,0,0,0]  (7 bit)
        """
        k = n.bit_length() - 1          # floor(log2(n))
        bits = [0] * k                  # k buah bit nol sebagai unary prefix
        for shift in range(k, -1, -1):  # k+1 bit nilai biner, MSB lebih dulu
            bits.append((n >> shift) & 1)
        return bits

    @staticmethod
    def _pack_bits(bits):
        """
        Mengemas list of bits ke dalam sebuah bytes object, dengan menambahkan
        zero-padding di sebelah kanan hingga batas byte terdekat.
        """
        padding = (8 - len(bits) % 8) % 8
        padded = bits + [0] * padding
        result = bytearray()
        for i in range(0, len(padded), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | padded[i + j]
            result.append(byte)
        return bytes(result)

    @staticmethod
    def _unpack_bits(data):
        """Membuka kemasan bytes object menjadi list of bits (MSB lebih dulu per byte)."""
        bits = []
        for byte in data:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
        return bits

    @staticmethod
    def eg_encode(list_of_numbers):
        """
        Melakukan Elias-Gamma encoding terhadap list of non-negative integers.

        Menerapkan offset +1 agar nilai 0 dapat direpresentasikan (Elias-Gamma
        hanya terdefinisi untuk n >= 1). Menambahkan header 4-byte (big-endian)
        yang berisi jumlah bilangan yang di-encode, agar decoding tetap benar
        meskipun ada zero-padding di akhir bit stream.

        Parameters
        ----------
        list_of_numbers: List[int]
            List of non-negative integers yang akan di-encode.

        Returns
        -------
        bytes
            Header 4-byte jumlah elemen diikuti packed bit stream.
        """
        bits = []
        for n in list_of_numbers:
            bits.extend(EliasGammaPostings._encode_number_bits(n + 1))
        header = struct.pack('>I', len(list_of_numbers))
        return header + EliasGammaPostings._pack_bits(bits)

    @staticmethod
    def eg_decode(encoded_bytestream):
        """
        Melakukan decoding sebuah bytestream yang sebelumnya di-encode dengan
        Elias-Gamma encoding.

        Membaca header 4-byte untuk mengetahui jumlah bilangan, kemudian
        men-decode sejumlah itu dari packed bit stream, dengan membalik offset +1.

        Parameters
        ----------
        encoded_bytestream: bytes
            Keluaran dari eg_encode.

        Returns
        -------
        List[int]
            List of non-negative integers hasil decoding.
        """
        count = struct.unpack('>I', encoded_bytestream[:4])[0]
        bits = EliasGammaPostings._unpack_bits(encoded_bytestream[4:])
        numbers = []
        i = 0
        while len(numbers) < count:
            # Hitung bit nol di depan untuk mendapatkan k = floor(log2(n))
            k = 0
            while bits[i] == 0:
                k += 1
                i += 1
            # Baca k+1 bit untuk mendapatkan nilai biner n
            n = 0
            for _ in range(k + 1):
                n = (n << 1) | bits[i]
                i += 1
            numbers.append(n - 1)  # balik offset +1 yang diterapkan saat encoding
        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes menggunakan Elias-Gamma
        Encoding berbasis gap. JANGAN LUPA diubah dulu ke gap-based list,
        sebelum di-encode.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings) yang sudah terurut.

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
            setelah dikompresi dengan Elias-Gamma Encoding.
        """
        gap_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_list.append(postings_list[i] - postings_list[i - 1])
        return EliasGammaPostings.eg_encode(gap_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes menggunakan
        Elias-Gamma Encoding.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies (selalu >= 1, tidak perlu konversi gap).

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di
            setiap dokumen pada list of postings, setelah dikompresi.
        """
        return EliasGammaPostings.eg_encode(tf_list)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode masih berupa gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        gap_list = EliasGammaPostings.eg_decode(encoded_postings_list)
        postings = [gap_list[0]]
        for i in range(1, len(gap_list)):
            postings.append(postings[-1] + gap_list[i])
        return postings

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes.

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai
            keluaran dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari
            encoded_tf_list.
        """
        return EliasGammaPostings.eg_decode(encoded_tf_list)


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
