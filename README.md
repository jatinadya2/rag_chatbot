# RAG Chatbot for Machine Manuals

This repository provides an advanced Retrieval-Augmented Generation (RAG) chatbot tailored for interacting with complex technical PDFs, such as machine operation and maintenance manuals. The system leverages a multi-stage pipeline—including sophisticated PDF preprocessing—to ensure highly accurate retrieval and answer generation.

## Overview

The chatbot is designed to:
- Parse, extract, and structure technical content from machine manuals in PDF format.
- Enable natural language question-answering over the entirety of manual content (including text, images, tables, and diagrams).
- Deliver robust, contextually relevant answers with references to specific sections or visuals.

## Why PDFs Need Preprocessing Before RAG

### 1. PDF Documents Are Not Readily Searchable
- **Non-linear Structure**: PDFs often encode paragraphs, tables, and images as unstructured byte streams, unlike HTML or Markdown. Text may be split across columns or scattered based on visual layout, complicating direct extraction.
- **Images and Tables**: Essential information is often embedded in figures/tables as images or as structured data that's not easily accessible with plain text extraction.

### 2. Accurate Retrieval Requires Clean, Meaningful Chunks
- **Text Chunking**: For RAG to retrieve relevant context, raw PDF content must be split into coherent, context-rich segments (chunks). Preprocessing identifies logical breakpoints, such as section headers and paragraphs.
- **Table Normalization**: Extracted tables must be reconstructed into structured formats (e.g., Markdown tables), correcting for OCR/text parsing errors, to allow queries over tabular data.
- **Image Captioning**: Diagrams and figures need machine-generated captions and context-aware metadata, which allow them to be referenced in answers.

### 3. Removing Noise and Duplicates
- **Redundant Content**: Manuals contain repetitive warnings, legal disclaimers, and duplicated icons/logos. Preprocessing filters these out using image hashing and textual heuristics.
- **Language & OCR Correction**: Many PDFs are scans or multi-language and require OCR plus intelligent cleaning to prevent retrieval of irrelevant or garbled content.

## Key Features

- **Multi-stage PDF Extraction**: Uses modern libraries (PyMuPDF, pdfplumber, Camelot, pytesseract) for robust parsing of text, tables, and images.
- **Heuristic and NLP-Based Structuring**: Segments text by headings, formats tables/rows intelligently, and tags special instructions (e.g., warnings or notes).
- **Image & Diagram Handling**: Extracts and deduplicates figures, then generates GPT-based captions contextualized by nearby text.
- **Flexible RAG Index**: After preprocessing, all extracted assets are embedded and indexed (using Pinecone) for lightning-fast, semantic retrieval.
- **Conversational Memory**: The chatbot maintains context across turns, supporting step-by-step troubleshooting and follow-up queries.

## Repository Structure

| File / Folder                | Purpose                                                      |
|------------------------------|--------------------------------------------------------------|
| `extract_pdf_pass1()`        | Initial pass: extracts and saves text, tables, images.       |
| `extract_pdf_pass2()`        | Second pass: refines structure, captions images, formats tables. |
| `data/`                      | Stores extracted data assets (text chunks, images, tables).  |
| `final_image_captions.json`  | Caches generated image captions for fast lookup.             |
| `main notebook (.ipynb)`     | End-to-end orchestration and experimentation.                |

## Workflow

1. **PDF Ingestion and Preprocessing**
    - Chunk and normalize all sections, resolving headings and hierarchy.
    - Extract tables and reformat for structure—even using OCR fallback.
    - Deduplicate and caption images for reference in both retrieval and answers.

2. **Index Construction**
    - Embed all document pieces (text, image captions, tables) into a semantic vector store.
    - Attach metadata (page, section, figure/table labels) for better filtering and grounding.

3. **Question Answering**
    - Retrieve the most relevant document segments using dense retrieval.
    - Augment LLM prompts with structured context—prioritizing tables, diagrams, or instructions as needed.

4. **Conversational Experience**
    - Maintain conversation history to allow iterative, multi-step support.

## Example Usage

```python
# Example: Ask about safety procedures for the spindle
ask("What safety devices are present for the spindle?")

# Example: Request a diagram or table
ask("Show me Figure 5 and Table B referenced in the section about speed adjustment.")
```

## Requirements

- Python 3.9+
- PyMuPDF, pdfplumber, Camelot, Tesseract, spaCy (en_core_web_sm), OpenAI SDK, Pinecone SDK, etc.

## When to Use This Chatbot

- Interactive support/help desk for machine operators and technicians.
- Onboarding or training for new staff—extracts structured insights from lengthy manuals.
- Quick search over large, image- and table-heavy technical manuals where traditional PDF search fails.

## FAQ

**Q:** _Why not just load raw PDF into RAG?_

**A:** Raw PDFs are noisy, unstructured, and often miss key data (tables/images) unless preprocessing is performed. Preprocessing ensures context is meaningful, chunks are searchable, and diagrams/tables are not ignored.

**Q:** _Does preprocessing increase accuracy?_

**A:** Yes. Clean, logically structured data allows the retrieval model to surface exact answers—rather than missing, irrelevant, or duplicated content.

## Credits & License

- Developed using open-source Python and LLM libraries.
- For academic, research, or non-commercial use—consult repository license for details.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/84246170/4392bbe1-2c74-4357-91a8-b1a6b285f4b0/rag_chatbot_structured.ipynb