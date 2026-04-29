# Multi-Hop RAG on HotpotQA

Benchmarking retrieval strategies for multi-hop QA on the HotpotQA hard subset (200 questions).

---

## How it works

Each retrieval method runs a **3-hop loop**:
1. LLM generates queries from the question → passages retrieved
2. LLM refines queries using hop-1 findings → more passages retrieved
3. LLM merges all retrieved passages + predicted supporting facts → final answer

The difference between methods is purely the **retrieval engine** underneath this loop.

---

## Results

| Method | Answer EM | Answer F1 | SF EM | SF F1 |
|---|---|---|---|---|
| No context | 0.274 | 0.346 | 0.000 | 0.000 |
| Oracle context — 1 hop | 0.502 | 0.680 | 0.433 | 0.735 |
| Oracle context — 2 hop | 0.557 | 0.674 | 0.348 | 0.687 |
| Hybrid + reranking | 0.443 | 0.573 | 0.184 | 0.359 |
| Hybrid (no reranking) | 0.408 | 0.550 | 0.169 | 0.353 |
| SentenceTransformers | 0.388 | 0.498 | 0.134 | 0.290 |
| Pyserini (BM25) | 0.353 | 0.489 | 0.124 | 0.301 |

**SF** = Supporting Facts &nbsp;|&nbsp; **EM** = Exact Match

### Key findings

- **Retrieval gap to oracle is ~0.11 F1** — retrieval quality is the primary bottleneck, not reasoning
- **Reranking adds ~3.5 EM / ~2.3 F1** over no-reranking — meaningful signal from cross-encoder reranking
- **2-hop reasoning over oracle context beats 1-hop** (EM: 0.557 vs 0.502) — iterative reasoning helps even with perfect context
- **Dense retrieval alone underperforms hybrid** — SentenceTransformers trails hybrid+rerank by ~7.5 F1 points

---

## Methods

| Method | Retrieval engine |
|---|---|
| Pyserini | BM25 sparse retrieval |
| SentenceTransformers | BGE dense embeddings + ChromaDB |
| Hybrid (no reranking) | BM25 + dense score fusion |
| Hybrid + reranking | BM25 + dense + cross-encoder reranker |
| Oracle context | Gold passages + distractors handed directly (no retrieval) |

---

## Stack

`Python` `LangChain` `ChromaDB` `Pyserini` `SentenceTransformers` `Claude API` `HotpotQA`

---

## Author

**Vignesh S** — Backend engineer (3 yrs) turned ML practitioner  
[GitHub](https://github.com/srikrishnavignesh) · [LinkedIn](https://www.linkedin.com/in/vignesh-suresh-796b83193/)
