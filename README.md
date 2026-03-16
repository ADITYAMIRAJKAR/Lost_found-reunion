#  Lost & Found Reunion – Multi-Modal Semantic Search

This project implements a **multi-modal semantic search engine for lost items**.  
It allows users to search for lost products using natural language descriptions and retrieves the most similar items based on **semantic similarity of text embeddings**.

The system combines **web scraping, embeddings, vector search, and AI models** to find the most relevant matches.

---
# Problem Statement

Students frequently lose items such as phones, laptops, tablets, and headphones.

Lost & Found offices log items using vague descriptions, making it difficult to identify the correct owner.

Problems include:

* vague descriptions
* manual search through spreadsheets
* no image search
* items often get donated before the owner finds them

This project solves these issues using **AI semantic search**.

---
## 2. Approach Summary

The project is built as an end-to-end pipeline with clear phases:

1. Data sourcing:
- Fetch products from DummyJSON.
- Download multiple images per item.
- Store metadata in CSV.

2. Data enrichment and cleaning:
- Generate realistic "lost item" narratives.
- Normalize categories.
- Remove duplicate/noisy records.
- Build two text fields:
  - `search_text` for rich metadata,
  - `clip_text` for concise CLIP-friendly semantics.

3. Embeddings and indexing:
- Use CLIP (`openai/clip-vit-base-patch32`) for both text and images.
- Encode all images per item and average vectors.
- Build 3 FAISS indexes:
  - text index,
  - image index,
  - multimodal (fused) index.

4. Retrieval and explainability:
- Route query to the correct index based on mode.
- Score and rank results with confidence.
- Generate explanations with Ollama (fallback to deterministic rule-based explanation).

5. Delivery:
- FastAPI backend.
- Streamlit UI for quick demo and interaction.

## 3. Architecture

```text
User (Streamlit UI)
    -> FastAPI /search endpoints
        -> SearchEngine
            -> CLIP text/image encoders
            -> FAISS indexes (text / image / multimodal)
            -> Metadata lookup
            -> Ollama explanation (with fallback)
```

## 4. Why CLIP for Multi-Modal Search

CLIP places text and image embeddings in a shared space, enabling:
- text-to-image matching,
- image-to-image retrieval,
- text+image fusion for stronger intent alignment.

This avoids separate incompatible embedding spaces and makes cross-modal retrieval practical.



## 5. Explainability with Ollama

The system uses local Ollama for natural-language explanations:
- endpoint: `POST /api/generate`
- model: configurable (default `llama3.2:3b`)
- timeout: configurable
- automatic fallback: deterministic explanation if Ollama is unavailable.

### Why Ollama

Ollama was chosen for this project because it is practical for campus/demo environments:
- runs fully local (no external paid API required),
- keeps search context/data on local machine,
- easy to set up and switch models quickly,
- works well with FastAPI for low-friction integration,
- provides controllable latency and predictable offline behavior.

### Model Used

Primary model used:
- `llama3.2:3b`

Reason:
- good quality-to-speed balance on normal laptops,
- fast enough for per-result explanation generation in a demo app,
- lightweight compared to larger local models.

Alternative options (if hardware allows):
- `mistral:7b` for stronger language quality (slower),
- `llama3.1:8b` for better reasoning (heavier, more RAM/VRAM needed).

Performance optimization:
- only top-N results use Ollama (`OLLAMA_EXPLAIN_TOP_K`),
- remaining results use deterministic explanation for speed.

This is a complete, demo-ready foundation that can be extended to real campus Lost & Found data.
