# Multi-Hop RAG Agent — HotpotQA

> Benchmarking retrieval strategies for complex multi-hop question answering using ChromaDB, BGE embeddings, BM25, and iterative LLM-guided query refinement.

---

## Overview

HotpotQA is a multi-hop QA benchmark where answering a question requires connecting information across **two separate Wikipedia passages**. This project builds and benchmarks an end-to-end RAG pipeline specifically designed for this multi-hop reasoning challenge.

All evaluations are conducted on the **hard subset (200 questions)** — the most challenging split where supporting passages are semantically distant and not directly retrievable by simple keyword matching.

---

## Results

| Method | Answer EM | Answer F1 | Supporting Facts EM | Supporting Facts F1 |
|---|---|---|---|---|
| No Context (LLM only) | 0.274 | 0.349 | 0.000 | 0.000 |
| BM25 Multi-Hop (Pyserini) | 0.363 | 0.480 | 0.075 | 0.253 |
| Dense Multi-Hop (BGE + ChromaDB) | 0.388 | 0.521 | 0.119 | 0.332 |
| Golden Context — Single Hop | 0.478 | 0.646 | 0.458 | 0.754 |
| Golden Context — Multi Hop | **0.527** | **0.669** | 0.428 | 0.742 |

**Key takeaway:** Dense retrieval (0.52 F1) outperforms BM25 (0.48 F1) on semantic multi-hop reasoning. A 0.15 F1 retrieval quality gap remains vs golden context — targeted by the fine-tuning stage.

---

## Pipeline

```
Question
   │
   ▼
Round 1 — Retrieve top-k passages (BGE + ChromaDB)
   │
   ▼
LLM — Identify supporting facts, generate context_needed queries
   │
   ▼
Round 2/3 — Retrieve with enriched queries
   │
   ▼
LLM — Final answer with full context
```

### Why Iterative Retrieval

HotpotQA questions require two hops:
- **Hop 1** — directly related to the question
- **Hop 2** — only reachable after finding hop 1

A single retrieval pass frequently misses hop 2. Iterative LLM-guided refinement enriches the query with hop 1 evidence before retrieving hop 2.

---

## Methods Benchmarked

### 1. No Context
Baseline — LLM answers from parametric knowledge alone. No retrieval.

### 2. Single Hop with Golden Context
Upper bound — golden supporting passages provided directly. Measures LLM reasoning capability independent of retrieval.

### 3. Multi Hop with Golden Context
Golden context + iterative LLM reasoning across 3 hops. Beats single hop golden by **0.05 F1** — showing iterative reasoning helps even with perfect context.

### 4. BM25 Multi-Hop (Pyserini)
Sparse retrieval using BM25. Strong for direct keyword/named-entity queries. Weak for semantic multi-hop reasoning where the bridge between hops is implicit.

### 5. Dense Multi-Hop (BGE + ChromaDB)
Dense retrieval using `BAAI/bge-base-en` embeddings stored in ChromaDB. Outperforms BM25 on semantic reasoning. Struggles with short named-entity queries ("Who is X?") where BM25 has an edge.

---

## Chunking Strategy

```python
# Split by sentences (pre-segmented in corpus)
# Max ~340 words per chunk (512 tokens / 1.5)
# 1 sentence overlap between chunks
# NLTK sent_tokenize for sentences exceeding word limit
```

Each chunk is stored with `title:` prefix to aid multi-hop title-based retrieval — critical for HotpotQA where article titles are the bridge between hops.

---

## Retrieval Design

**Indexing:**
```python
embedding = bge_model.encode("passage: " + chunk, normalize_embeddings=True)
```

**Querying:**
```python
embedding = bge_model.encode("query: " + question, normalize_embeddings=True)
```

BGE uses asymmetric prefixes (`query:` vs `passage:`) — using the wrong prefix at query time significantly degrades results.

---

## LLM Prompt Design

The LLM is prompted with:
- `question` — the original question
- `context` — retrieved passages organized by title and sentence id
- `provide_answer` — whether to answer or only identify supporting facts
- `prev_resp_comments` — feedback from previous iteration for self-correction

Output:
- `answer` — verbatim from context
- `supporting_facts` — titles and sentence ids used for reasoning
- `context_needed` — context-rich queries for next retrieval round

Context-needed queries are declarative and passage-style rather than question-style to better match the embedding space:
```
Good: "Townsend Coleman American voice actor Teenage Mutant Ninja Turtles"
Bad:  "Who is Townsend Coleman?"
```

---

## Key Findings

- **Dense > BM25** for semantic multi-hop reasoning (+0.04 F1)
- **BM25 > Dense** for direct named-entity lookups — motivates hybrid retrieval
- **Iterative reasoning** improves over single pass even with golden context (+0.05 F1)
- **Supporting facts F1 gap** (0.33 retrieved vs 0.75 golden) indicates retrieval quality is the primary bottleneck, not LLM reasoning
- **Hard questions only** — results are conservative; easy/medium questions would push numbers higher

---

## Retrieval Quality Gap

```
Dense retrieval Answer F1:    0.521
Golden context Answer F1:     0.669
Gap:                          0.148
```

This gap motivates **contrastive fine-tuning** of BGE on HotpotQA triplets:

```python
{
  "query":    "question: ...",
  "positive": "passage: supporting_fact",
  "negative": "passage: distractor"
}
```

Hop 2 training uses query enrichment:
```python
"query: {question} evidence: {hop1_supporting_fact}"
```

---

## Stack

| Component | Tool |
|---|---|
| Vector DB | ChromaDB |
| Embeddings | BAAI/bge-base-en |
| Sparse Retrieval | Pyserini (BM25) |
| LLM | — |
| Dataset | HotpotQA |
| Training (planned) | sentence-transformers + MultipleNegativesRankingLoss |

---

## Planned

- Contrastive fine-tuning of BGE on HotpotQA triplets
- Hybrid retrieval (BM25 + dense reranking) to combine strengths of both
- Evaluation on easy + medium subsets for complete benchmark

---

## Dataset

[HotpotQA](https://hotpotqa.github.io/) — Yang et al., 2018. Multi-hop question answering dataset requiring reasoning over two Wikipedia passages with supporting fact annotations.
