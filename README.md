# RAG Chatbot over Heterogeneous PDFs

> **TL;DR**  
> This notebook ingests a PDF, restructures it into consistent, metadata‑rich chunks, embeds those chunks, stores them in a vector DB, and serves them through a lightweight Q&A chatbot. The README explains **why the structuring step is mandatory**, how the notebook does it, and how to run / extend the pipeline.

---

## Table of Contents

1. [Why Every PDF Must Be Structured Before Embedding](#why-structure)
2. [End‑to‑End Architecture](#architecture)
3. [Notebook Walkthrough (Section by Section)](#walkthrough)
4. [Data Modeling & Chunk Schemas](#schemas)
5. [How to Run This Project](#run)
6. [Project/Folder Layout](#layout)
7. [Evaluation & Debugging Tips](#eval)
8. [Common Errors & Fixes](#troubleshooting)
9. [Roadmap / Next Steps](#roadmap)
10. [License](#license)

---

<a id="why-structure"></a>
## 1. Why Every PDF Must Be Structured Before Embedding

Raw PDFs are **not** uniform:
- **Layouts differ**: single vs multi‑column, headings vs sidebars, footers/headers, callout boxes.  
- **Mixed content types**: paragraphs, tables, figures, scanned images (OCR needed), equations.  
- **Noise & boilerplate**: page numbers, disclaimers, repeated headers/footers, legal text.  
- **Semantics lost in extraction**: a table row, a caption, and a paragraph can be smashed into one flat string if you just `pdf_text()` and call it a day.  
- **Granularity matters**: Embeddings work best when chunks are semantically coherent. A 4‑page blob produces muddy vectors; a well‑scoped, labeled chunk returns precise hits.

**If you skip structuring**, you get:
- Embeddings of noisy text → lower retrieval precision.
- Harder metadata filtering → you can’t target “only tables” or “only 2018 reports”.
- Garbage in ↔ garbage out: LLM hallucinations increase when context is off.

**So we**:
1. **Extract each modality separately** (text, tables, images via OCR if needed).  
2. **Normalize & clean** (strip headers/footers, fix Unicode, collapse whitespace).  
3. **Segment** into logical units (section, subsection, table row, figure caption).  
4. **Annotate with metadata** (doc_id, page, section path, content_type, source_pdf).  
5. **Chunk smartly** (token budget aware, sliding windows, preserve table structure).  
6. **Embed & store** (vector DB + metadata).  
7. **Retrieve with filters + reranking** -> feed LLM.

---

<a id="architecture"></a>
## 2. End‑to‑End Architecture

### 2.1 High-Level Flow

```mermaid
graph TD
    A[Raw PDFs] --> B[Ingestion & Parsing]
    B --> C[Cleaning & Structuring]
    C --> D[Chunking + Metadata]
    D --> E[Embeddings Model]
    E --> F[Vector Store (e.g., Pinecone)]
    F --> G[Retriever]
    G --> H[LLM Prompt Builder]
    H --> I[Chatbot / API / UI]

### 2.2 Query-Time Sequence

sequenceDiagram
    participant U as User
    participant R as Retriever
    participant VS as Vector Store
    participant LLM as LLM
    U->>R: Ask question
    R->>VS: similarity_search / MMR (k docs)
    VS-->>R: docs + scores
    R->>LLM: Build prompt with top chunks
    LLM-->>U: Final answer + cited chunks



