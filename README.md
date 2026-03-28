# Tugas Pemrograman 2 TBI: Search Engine from Scratch

A text search engine built entirely from scratch using only Python's standard library. This project implements the full indexing and retrieval pipeline, from raw document text to ranked search results, without relying on any ready-made search engine library.

The collection contains **1,033 documents** evaluated against **30 queries** with human-annotated relevance judgments (qrels). The system is comparable in scope to libraries such as [PyTerrier](https://pyterrier.readthedocs.io/) and [Pyserini](https://github.com/castorini/pyserini).

---

## Project Structure

```
.
├── bsbi.py                 # Core indexing (BSBI) and retrieval logic (TF-IDF, BM25, WAND)
├── compression.py          # Postings encoding: Standard, VBE, Elias-Gamma
├── index.py                # Low-level inverted index reader/writer
├── util.py                 # IdMap and postings merge utilities
├── preprocessing.py        # Tokenization, stopword removal, stemming (Porter)
├── evaluation.py           # Evaluation runner: RBP, DCG, NDCG, AP over 30 queries
├── search.py               # Example retrieval script for sample queries
│
├── spimi.py                # Bonus: SPIMI indexing (alternative to BSBI)
├── trie.py                 # Bonus: Trie data structure for autocomplete
├── spell_correction.py     # Bonus: Levenshtein-based spell correction
├── lsi.py                  # Bonus: Latent Semantic Indexing (Truncated SVD)
├── query_expansion.py      # Bonus: Pseudo-Relevance Feedback (Rocchio)
├── adaptive_reranking.py   # Bonus: Adaptive Re-ranking with Corpus Graph
├── snippets.py             # Bonus: KWIC document snippet generation
├── interactive_search.py   # Bonus: Interactive search REPL (all modes)
│
├── collection/             # Document corpus (1,033 text files)
├── index/                  # Generated binary index files (produced by bsbi.py)
├── tmp/                    # Temporary intermediate indices (used during indexing)
├── qrels.txt               # Query relevance judgments (human-annotated)
└── queries.txt             # 30 evaluation queries
```

---

## Setup

### 1. Install dependency

```bash
pip install tqdm
```

### 2. Build the index

```bash
python bsbi.py
```

This scans all documents in `collection/`, builds intermediate inverted indices block by block using the BSBI algorithm, then merges them into a single index in `index/`. The `index/` and `tmp/` directories must exist before running.

### 3. Run a search

```bash
python search.py
```

Runs three sample queries and prints top-10 results for both TF-IDF and BM25.

### 4. Evaluate search quality

```bash
python evaluation.py
```

Evaluates TF-IDF and BM25 across 30 queries using four metrics (RBP, DCG, NDCG, AP).

### 5. Start the interactive search REPL

```bash
python interactive_search.py
```

Launches an interactive CLI with all retrieval modes, autocomplete, spell correction, and document snippets. See the [Interactive Search REPL](#interactive-search-repl) section for full details.

---

## Mandatory Features (100 pts)

### 1. Elias-Gamma Bit-Level Compression (`compression.py`)

Added `EliasGammaPostings` alongside the existing `VBEPostings`. While VBE operates at the byte level, Elias-Gamma compresses at the bit level and is particularly efficient for small gap values (dense postings lists).

**How it works:**

Postings lists are gap-encoded before compression (same as VBE). Each gap value `n` is encoded as:
```
floor(log2(n)) zero bits  +  floor(log2(n))+1 bits of binary representation of n
```
A `+1` offset is applied so that gap values of 0 can be handled safely (Elias-Gamma is only defined for `n >= 1`). The bytestream is prefixed with a 4-byte big-endian count header so the decoder knows exactly how many integers to read regardless of zero-padding at the end of the last byte.

**To use Elias-Gamma instead of VBE**, change the encoding parameter when building the index:

```python
# In bsbi.py __main__ block:
BSBIIndex(data_dir='collection', postings_encoding=EliasGammaPostings, output_dir='index')
```

**Size comparison for postings list `[34, 67, 89, 454, 2345738]`:**

| Method      | Postings | TF list  |
|-------------|----------|----------|
| Standard    | 20 bytes | 20 bytes |
| VBE         | 9 bytes  | 5 bytes  |
| Elias-Gamma | 16 bytes | 8 bytes  |

Elias-Gamma is most efficient for very small gap values (gap <= 7 fits in <= 5 bits vs. VBE's 1-byte minimum), making it ideal for dense postings lists. Compression is fully lossless and does not affect retrieval accuracy.

---

### 2. BM25 Scoring (`bsbi.py`, function `retrieve_bm25`)

Added `retrieve_bm25()` to `BSBIIndex` using the Okapi BM25 formula:

```
IDF(t)   = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

TF_norm  = tf(t, D) * (k1 + 1)
           ------------------------------------------
           tf(t, D) + k1 * (1 - b + b * |D| / avgdl)

Score(D, Q) = sum over t in Q of IDF(t) * TF_norm(t, D)
```

Default parameters: `k1 = 1.2`, `b = 0.75`.

Document lengths (`|D|`) and average document length (`avgdl`) are pre-computed during indexing and stored in `doc_length` with no extra overhead at query time. Retrieval follows a Term-at-a-Time (TaaT) strategy identical to the existing TF-IDF method.

**Effect on retrieval accuracy** (evaluated over 30 queries, `python evaluation.py`):

| Method | RBP    | DCG    | NDCG   | AP     |
|--------|--------|--------|--------|--------|
| TF-IDF | 0.6470 | 5.7773 | 0.7678 | 0.4938 |
| BM25   | 0.6698 | 5.8878 | 0.7823 | 0.5229 |

BM25 improves AP by **+5.9%**, NDCG by **+1.9%**, and RBP by **+3.5%** over TF-IDF.

---

### 3. Evaluation Metrics: DCG, NDCG, AP (`evaluation.py`)

Added three new metrics alongside the existing RBP metric. All take a binary relevance vector and the total number of relevant documents in the collection as input.

**Discounted Cumulative Gain (DCG):**
```
DCG = sum_{i=1}^{n}  rel_i / log2(i + 1)
```
Rewards relevant documents appearing at higher ranks by applying a logarithmic position discount.

**Normalized DCG (NDCG):**
```
NDCG = DCG / IDCG
```
IDCG is the DCG of the ideal ranking (all relevant documents placed at the top). NDCG is always in `[0, 1]`.

**Average Precision (AP):**
```
AP = (1 / R) * sum_{k=1}^{n}  P(k) * rel(k)
```
`R` is the total number of relevant documents in the collection, and `P(k)` is precision at rank `k`. AP is sensitive to the exact position of each relevant document and rewards early retrieval.

**To reproduce:**
```bash
python evaluation.py
```

Sample output:
```
Hasil evaluasi TF-IDF terhadap 30 queries
RBP  score = 0.6470
DCG  score = 5.7773
NDCG score = 0.7678
AP   score = 0.4938

Hasil evaluasi BM25 terhadap 30 queries
RBP  score = 0.6698
DCG  score = 5.8878
NDCG score = 0.7823
AP   score = 0.5229
```

---

### 4. WAND Top-K Retrieval (`bsbi.py`, function `retrieve_bm25_wand`)

Added `retrieve_bm25_wand()`, an efficient top-K retrieval algorithm that avoids fully evaluating every document in the postings lists. WAND maintains a score threshold `theta` equal to the K-th best score found so far and prunes any document whose maximum possible BM25 score cannot exceed `theta`.

**Per-term upper bound:**
```
UB[t] = IDF(t) * max_tf_t * (k1 + 1) / (max_tf_t + k1 * (1 - b))
```
`max_tf_t` (the maximum TF of term `t` across all postings) is stored in the index as the 5th element of `postings_dict[term]`.

**Algorithm per iteration:**
1. Sort active query terms by their current doc pointer.
2. Find the **pivot**: the first term `p` where the cumulative sum of upper bounds exceeds `theta`.
3. If no pivot exists, all remaining documents are pruned and the algorithm terminates.
4. Let `pivot_doc = terms[p].current_doc`:
   - If `terms[0].current_doc == pivot_doc`: fully evaluate `pivot_doc` with exact BM25, update the min-heap of top-K and `theta`, then advance all pointers at `pivot_doc`.
   - Otherwise: skip all terms before `p` directly to `pivot_doc` using binary search (`bisect_left`).

**WAND produces identical top-K results as BM25**. Accuracy is unchanged and speed improves by skipping provably non-competitive documents.

---

## Bonus Features (150 pts)

### 1. SPIMI Indexing (`spimi.py`)

Implemented Single-Pass In-Memory Indexing (SPIMI) as an alternative to BSBI. Unlike BSBI, SPIMI does not require a global Term-ID mapping: it processes each block entirely in memory, sorts terms as strings, and writes partial indices per block before merging.

**To build the index using SPIMI:**
```bash
python spimi.py
```

---

### 2. Trie-based Autocomplete (`trie.py`)

Implemented a Trie (prefix tree) built from the full index vocabulary, with each node storing the document frequency of the corresponding term. Autocomplete returns up to `top_k` completions ranked by document frequency (most common terms first).

The Trie is serialized to `index/trie.pkl` on first use and loaded from disk on subsequent runs.

**To test directly:**
```python
from trie import Trie
trie = Trie.load('index/trie.pkl')
print(trie.autocomplete('lip', top_k=5))
```

**In the REPL:** use `:ac <prefix>` (see [Interactive Search REPL](#interactive-search-repl)).

---

### 3. Spell Correction (`spell_correction.py`)

Implemented `SpellCorrector` using Levenshtein edit distance. For each out-of-vocabulary query term, it finds the closest vocabulary term within a configurable maximum edit distance (default: 2).

The corrector is applied automatically to each query in the REPL. If any term is corrected, the REPL prints the original and corrected query before retrieval. Correction can be toggled on/off with `:spell off`.

---

### 4. Latent Semantic Indexing (`lsi.py`)

Implemented `LSIRetriever` using Truncated SVD on the TF-IDF term-document matrix. The matrix is decomposed into `k=100` latent dimensions and queries are projected into the same reduced space for cosine similarity retrieval.

**Algorithm:**
1. Build TF-IDF term-document matrix `A` (|V| x N).
2. Apply Truncated SVD: `A ≈ U_k · Σ_k · V_k^T`
3. Project query: `q_k = Σ_k^{-1} · U_k^T · q_tfidf`
4. Rank documents by cosine similarity between `q_k` and columns of `V_k`.

The LSI model is saved to `index/lsi_model.pkl` after fitting.

**To fit and evaluate:**
```bash
python lsi.py
```

**Effect on MAP** (evaluated over 30 queries, `k=1000`):

| Method | MAP    | vs BM25  |
|--------|--------|----------|
| BM25   | 0.5229 | baseline |
| LSI    | 0.6758 | +29.2%   |

LSI captures semantic similarity beyond exact keyword matching, allowing it to retrieve relevant documents that do not share exact terms with the query.

---

### 5. Pseudo-Relevance Feedback (Rocchio) (`query_expansion.py`)

Implemented `RocchioQueryExpansion`, which expands the query using the centroid of the top-5 pseudo-relevant BM25 documents.

**Rocchio formula:**
```
q_new = alpha * q_orig + beta * centroid(top feedback docs)
```
Default: `alpha=1.0`, `beta=0.75`, `num_feedback_docs=5`, `num_expansion_terms=10`.

The top-10 new terms (by weight increase) are added to the query before re-retrieval with BM25.

**To evaluate:**
```bash
python query_expansion.py
```

**Effect on MAP** (evaluated over 30 queries, `k=1000`):

| Method   | MAP    | vs BM25 |
|----------|--------|---------|
| BM25     | 0.5229 | baseline |
| BM25+PRF | 0.5749 | +9.9%   |

---

### 6. Adaptive Re-ranking (`adaptive_reranking.py`)

Implemented `AdaptiveReranker`, inspired by MacAvaney et al. (2022) "Adaptive Re-Ranking with a Corpus Graph". The idea: documents that are similar in content to high-scoring BM25 documents tend to also be relevant. Score propagation through a document similarity graph re-ranks the initial BM25 results.

**Algorithm:**
1. Retrieve `initial_k=400` candidates from BM25.
2. Build TF-IDF vectors for each candidate document.
3. Build a corpus graph: add an edge between two documents if their cosine similarity exceeds `graph_threshold=0.05`.
4. Normalize BM25 scores to `[0, 1]`.
5. Propagate scores (one-step message passing):
   ```
   score_new[d] = (1 - alpha) * score_bm25[d]
                + alpha * Σ_{d'} sim(d, d') * score_bm25[d'] / (|N(d)| + 1)
   ```
6. Re-rank by `score_new`, return top-K.

Default parameters: `alpha=0.97`, `initial_k=400`, `graph_threshold=0.05`.

**To evaluate:**
```bash
python adaptive_reranking.py
```

**Effect on MAP** (evaluated over 30 queries, `k=1000`):

| Method           | MAP    | vs BM25 |
|------------------|--------|---------|
| BM25             | 0.5229 | baseline |
| Adaptive Re-rank | 0.6085 | +16.4%  |

---

### 7. Document Snippets (KWIC) (`snippets.py`)

Implemented `SnippetGenerator` using a Key Word In Context (KWIC) sliding window algorithm. For a given query, it finds the densest region of query term occurrences in the document and returns a ~35-word window around it as the snippet.

**Algorithm:**
1. Tokenize and stem the document and the query.
2. Mark hit positions: positions where a document token matches any query stem.
3. Use an O(N) sliding window to find the window of `window_size=35` words with the most hits.
4. Return the window text with `...` prefix/suffix if the window is not at the document boundary.

Snippets are displayed automatically in the REPL below each search result.

---

### 8. Interactive Search REPL (`interactive_search.py`)

A unified interactive CLI that integrates all components: six retrieval modes, spell correction, Trie autocomplete, and KWIC snippets.

**To start:**
```bash
python interactive_search.py
```

On startup, the REPL loads the index, Trie, spell corrector, LSI model (if available), and all retrievers. The default mode is BM25.

---

## Evaluation Results

All numbers evaluated over 30 queries with human-annotated relevance judgments.

**Base metrics** (`python evaluation.py`):

| Method | RBP    | DCG    | NDCG   | AP (= MAP) |
|--------|--------|--------|--------|------------|
| TF-IDF | 0.6470 | 5.7773 | 0.7678 | 0.4938     |
| BM25   | 0.6698 | 5.8878 | 0.7823 | 0.5229     |

**MAP comparison for advanced retrieval methods** (k=1,000, each vs BM25 baseline):

| Method               | MAP    | Improvement |
|----------------------|--------|-------------|
| BM25 (baseline)      | 0.5229 | baseline    |
| BM25 + WAND          | 0.5229 | identical (efficiency only) |
| BM25 + PRF (Rocchio) | 0.5749 | +9.9%       |
| Adaptive Re-ranking  | 0.6085 | +16.4%      |
| LSI (k=100)          | 0.6758 | +29.2%      |

---

## Interactive Search REPL

Start with `python interactive_search.py`. Type a query to search, or use commands prefixed with `:`.

### Retrieval modes

Switch with `:mode <name>`:

| Mode       | Description                                              |
|------------|----------------------------------------------------------|
| `tfidf`    | TF-IDF with Term-at-a-Time scoring                       |
| `bm25`     | Okapi BM25 (default)                                     |
| `wand`     | BM25 + WAND top-K pruning (same results, more efficient) |
| `lsi`      | Latent Semantic Indexing (requires `index/lsi_model.pkl`)|
| `prf`      | BM25 + Rocchio pseudo-relevance feedback                 |
| `adaptive` | BM25 + corpus graph re-ranking                           |

### Commands

| Command          | Description                                              |
|------------------|----------------------------------------------------------|
| `:mode <name>`   | Switch retrieval mode                                    |
| `:k <n>`         | Set number of results shown (default: 10)                |
| `:ac <prefix>`   | Show Trie autocomplete suggestions for a prefix          |
| `:spell on`      | Enable automatic spell correction (default: on)          |
| `:spell off`     | Disable automatic spell correction                       |
| `:help`          | Show all available commands                              |
| `:quit` / `:q`   | Exit the REPL                                            |

### Example session

```
[bm25] Query> lipid metabolism pregnancy
  1. [24.132] collection/1/7.txt
     ... lipid metabolism in normal and toxemic pregnancy was studied ...

[bm25] Query> :mode adaptive
  Mode diubah ke: ADAPTIVE

[adaptive] Query> lipid metabolism pregnancy
  1. [0.9821] collection/1/7.txt
     ... lipid metabolism in normal and toxemic pregnancy was studied ...

[adaptive] Query> :ac lip
  Autocomplete 'lip':
    lipid  (DF=87)
    lipoid (DF=12)
    lipopro (DF=9)

[adaptive] Query> :spell off
  Koreksi ejaan: NONAKTIF

[adaptive] Query> :k 5
  Top-K diubah ke: 5

[adaptive] Query> :quit
  Sampai jumpa!
```

---

## Conclusion

This project implements a complete information retrieval system from scratch, covering the full pipeline from indexing to ranked retrieval and interactive search. All four mandatory features are implemented and verified: Elias-Gamma bit-level compression, BM25 scoring, three additional evaluation metrics (DCG, NDCG, AP), and WAND top-K pruning.

Beyond the mandatory requirements, eight bonus features were added. Among the retrieval methods, LSI achieves the highest MAP improvement (+29.2% over BM25) by capturing semantic relationships between terms, while Adaptive Re-ranking (+16.4%) and Pseudo-Relevance Feedback (+9.9%) improve ranking by leveraging document similarity and query expansion respectively. The interactive REPL ties all components together into a single usable interface with real-time spell correction, autocomplete, and document snippets.

All evaluation numbers are reproducible by running the scripts listed in the Setup section above.