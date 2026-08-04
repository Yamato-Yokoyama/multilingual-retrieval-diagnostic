# 🌐 Multilingual Retrieval Diagnostic

### From Morphology to RAG — Why Tokenization Still Matters in 2026

**A portfolio project bridging Japanese, English, and German for cross-lingual search and recommendation systems.**

---

## The Question That Started This Project

Imagine you run an online shop in Germany. A customer searches for `Fahrradhelm` (bicycle helmet). Your database contains a product listed as `Kinderfahrradhelm` (children's bicycle helmet). A naive string match fails. Your customer leaves. You lose the sale.

Now imagine the reverse: the product is listed as `Fahrradhelm für Kinder`, and the customer searches for `Kinderfahrradhelm`. Same meaning. Same failure.

This isn't hypothetical. It happens every day on Zalando, Otto, Amazon.de — and the same class of problem exists in Japanese, where words are written **without spaces at all**. A product titled `扇風機60W省エネ静音リビング扇風機` needs to be matchable against a user typing `扇風機 リビング`.

**How do we build search systems that actually understand this?**

---

## "But Hasn't ChatGPT Solved Search Already?"

This was my first question. It's the wrong question.

Modern AI search products are built as a **stack**, and every layer depends on the one below it. When ChatGPT answers a question with sources, retrieval is running underneath. And retrieval quality depends on tokenization — how you split raw text into meaningful units.

> **"Your RAG is only as good as your retriever."**

Break the foundation, break everything above it. That's why this project starts at the bottom of the stack.

---

## Three Generations of Search

### 🔤 Generation 1 — Lexical Search (TF-IDF, BM25)

**Used by:** Elasticsearch, early Google, product search on most e-commerce sites.

- Splits text into tokens, counts word overlap between query and document
- Fast, transparent, strong on exact matches
- **Fails on:** synonym mismatch (`car` ≠ `automobile`), morphological variation (`Fahrradhelm` ≠ `Kinderfahrradhelm`), unseen compound forms

Tokenization is the whole foundation here. Split `Kinderfahrradhelm` wrong, and the inverted index can never find it.

### 🧠 Generation 2 — Semantic Search (Dense Embeddings)

**Used by:** modern Google, DeepL Search, any Vector DB stack (Pinecone, Weaviate, Qdrant).

- Encodes queries and documents into vectors that capture *meaning*
- `car` and `automobile` land near each other in vector space
- **Fails on:** exact identifiers (product codes, technical terms), long-tail entities, black-box explainability

Tokenization is still foundational — modern embedding models (BERT, E5, Sentence-Transformers) rely on **subword tokenization** internally (BPE, WordPiece, SentencePiece). If the subword split is meaningless, the embedding is meaningless.

### 🤖 Generation 3 — Generative Search / RAG

**Used by:** ChatGPT, Perplexity, Bing Chat, Google AI Overviews, and virtually every enterprise AI assistant being built in 2026.

- Retrieves documents (using Gen 1 + Gen 2)
- Feeds them into an LLM to generate a natural-language answer
- **Fails when retrieval fails** — the LLM will confidently hallucinate from wrong or missing documents

Production RAG at DeepL, SAP Joule, and Aleph Alpha lives or dies on retrieval quality. Retrieval quality lives or dies on tokenization quality.

---

## The Stack

```
┌─────────────────────────────────────┐
│  Generation 3 — RAG / Chat AI       │  ← 2026 products
├─────────────────────────────────────┤
│  Generation 2 — Semantic Search     │  ← Vector databases
├─────────────────────────────────────┤
│  Generation 1 — Lexical Search      │  ← TF-IDF, BM25
├─────────────────────────────────────┤
│  Foundation — Morphology &          │  ← This project starts here
│  Tokenization                       │
└─────────────────────────────────────┘
```

---

## Where NLP Actually Solves Problems in Search & Recommendation

The real, still-unsolved problems that companies are actively hiring for in 2026:

| # | Problem | Concrete Example | Where NLP Comes In |
|---|---------|------------------|--------------------|
| 1 | **Vocabulary mismatch** | User: `Auto`, Document: `PKW` | Embeddings, query expansion |
| 2 | **Morphological variation** | `Fahrradhelm` vs `Kinderfahrradhelm` | Compound splitting, lemmatization, subword tokenization |
| 3 | **Ambiguity** | `Apple` — fruit or company? | Contextual embeddings, NER |
| 4 | **Intent understanding** | `iPhone 15` — shopping or research? | Query classification |
| 5 | **Cross-lingual retrieval** | Japanese query, English documents | Multilingual embeddings (LaBSE, mE5) |
| 6 | **Cold-start recommendation** | New product with no interaction history | Content-based embeddings from titles/descriptions |
| 7 | **Long-tail queries** | Rare or novel search terms | Subword tokenization, LLM-based reformulation |

**Problem #2 is Week 1 of this project. Problem #5 is where this project is heading.**

---

## Why I'm Building This From the Ground Up

I'm a Computational Linguistics BA student at the University of Tübingen. My background is Japanese-native, English-fluent, currently learning German — which gives me a specific vantage point on this problem.

Japanese and German share a structural challenge that English does not have: writers pack morphemes together without spaces (Japanese) or into arbitrarily long compound words (German). English-centric NLP tools handle these languages poorly. Anyone building a genuinely multilingual product — DeepL, SAP Joule, Aleph Alpha, Zalando's search team — runs into this on day one.

Rather than jumping straight to fine-tuning a large model, I'm building from the foundation upward. When I discuss retrieval quality in an interview, I want to talk about it at the level of what is actually happening inside the pipeline.

The coursework I've been through at Tübingen and Temple gives me the raw materials:

- **SNLP1 / SNLP2** — classification, sequence labeling, feature engineering, evaluation
- **Text Technology** — Unicode, encoding, XML, structured data
- **Semantics / Pragmatics** — formal meaning, discourse, reference
- **Neural methods** — Word2Vec, GloVe, BERT/DistilBERT fine-tuning

This project is where I stitch those pieces together into something that solves a real problem.

---

## Project Roadmap — 4 Weeks

Each week produces a working artifact and a public writeup.

### 📅 Week 1 — Compound Word Detection *(Classification)*

- **Task:** Given a German word, classify whether it is a compound
- **Approach:** Character n-gram features + logistic regression (Scikit-learn)
- **Skill:** Feature engineering, binary classification, F1 evaluation
- **Deliverable:** `week1/` — training pipeline, evaluation notebook, LinkedIn writeup

### 📅 Week 2 — Compound Word Segmentation *(Sequence Labeling)*

- **Task:** Insert boundaries: `Kinderfahrradhelm` → `Kinder | Fahrrad | Helm`
- **Approach:** BIO tagging, sequence labeling
- **Skill:** The same core technique used in Japanese morphological analyzers (MeCab, Sudachi) and in Named Entity Recognition
- **Deliverable:** `week2/` — segmentation model, comparison against a Japanese tokenizer for parallel insight

### 📅 Week 3 — Document-Level Classification

- **Task:** Classify text by topic or author attribute
- **Approach:** GloVe embeddings weighted by TF-IDF → linear classifier / MLP
- **Skill:** Moving from token-level to document-level representation
- **Deliverable:** `week3/` — classification pipeline with confusion matrix and F1 analysis

### 📅 Week 4 — MVP: Retrieval + Ranking

- **Task:** Given a query, return the most relevant documents, ranked by relevance and quality
- **Approach:** TF-IDF + cosine similarity for retrieval; Week 3 classifier for ranking adjustment
- **Skill:** End-to-end retrieval pipeline
- **Deliverable:** `week4/` — MVP with clean documentation, demo, and roadmap toward RAG

---

## What Comes After Week 4

This 4-week project ends where the real research question begins: **multilingual retrieval for cross-lingual RAG**. Once the foundation is solid, the natural next steps are:

- Replace TF-IDF with multilingual dense embeddings (mE5, LaBSE)
- Extend the pipeline to Japanese ↔ English document pairs
- Build a small human-annotated benchmark to measure where cross-lingual retrieval breaks

The long-term goal: a diagnostic tool that shows *where* a cross-lingual retrieval system fails, and *why*.

---

## Repository Structure

```
multilingual-retrieval-diagnostic/
├── README.md              ← You are here
├── docs/
│   └── week-by-week/      ← Weekly reflection notes
├── week1/                 ← Compound detection
├── week2/                 ← Compound segmentation
├── week3/                 ← Document classification
├── week4/                 ← Retrieval MVP
├── data/
└── requirements.txt
```

---
## Data Sources

- **Week 1** (compound classification): SNLP1 course materials, University of Tübingen (2025)
- **Week 2** (compound segmentation): SNLP2 course materials, University of Tübingen, taught by Prof. Çağrı Çöltekin. Course-restricted dataset.

---
## About Me

BA student in Computational Linguistics (ISCL), University of Tübingen.
Previously: Computer Science at Temple University Japan, AI engineering internship at MetaMoJi, data analyst internship at FPT Software, LinkedIn Japan Student Ambassador.

Currently looking for **Werkstudent / Internship** positions in Applied AI Engineering — especially multilingual NLP, retrieval, and search — starting late August 2026.

- 📫 LinkedIn: *[https://www.linkedin.com/in/yamato-yokoyama/]*
- 💻 GitHub: *[https://github.com/Yamato-Yokoyama]*