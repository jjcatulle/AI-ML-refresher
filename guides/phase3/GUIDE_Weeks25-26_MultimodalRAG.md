# Guide: Multimodal RAG Prototype (Weeks 25-26)

## Beginner Start Here
This project extends classic RAG by adding non-text context like images, diagrams, and tables.

### What this project does
Build a retrieval + generation system that can answer questions using:
- text chunks,
- image/table captions,
- metadata links to source assets.

### Terms you must know first
- `Multimodal`: using more than one data modality (text, image, table).
- `Grounded answer`: response supported by retrieved evidence.
- `Citation correctness`: whether cited source actually supports the claim.
- `Recall@k`: percentage of queries where relevant evidence appears in top-k retrieval.
- `Faithfulness`: answer does not invent facts outside evidence.
- `Reranking`: second-stage model to improve top candidate ordering.

### Modules used
- `pandas`: metadata handling and evaluation tables.
- `langchain` or equivalent pipeline framework: retrieval and prompting.
- `sentence-transformers` (or API embeddings): text/image descriptor vectors.
- `faiss` / vector DB: similarity search.
- `rank-bm25`: keyword retrieval for hybrid search.

### How to study this guide
1. Build ingestion pipeline first.
2. Verify retrieval quality before generation quality.
3. Use strict citation checks.
4. Keep an explicit evaluation set and failure log.

---

## Big Picture
Most enterprise documents are multimodal: policy text, diagrams, screenshots, and tables.
A text-only retriever misses key evidence in visual content.

This project teaches you to combine modalities in one retrieval system and evaluate rigorously.

**Key Skills:**
- multimodal data preparation,
- hybrid retrieval and reranking,
- citation-grounded generation,
- retrieval-first evaluation strategy,
- cost/latency controls.

---

## Project Scope

### Inputs
- Markdown, PDF, and image assets.
- Optional OCR output for image text.

### Outputs
- Answer with evidence citations.
- Retrieved context bundle (text chunks + visual descriptors).
- Evaluation report (retrieval and answer metrics).

---

## Concept 1: Multimodal Ingestion

Create a unified schema for all retrievable units.
Each unit should include:
- `id`
- `modality` (text, image_caption, table_caption)
- `content`
- `source_doc`
- `section`
- `page_or_index`

```python
record = {
    "id": "doc3_p12_img2",
    "modality": "image_caption",
    "content": "Latency vs throughput chart comparing model versions",
    "source_doc": "architecture_review.pdf",
    "section": "Performance",
    "page_or_index": 12
}
```

---

## Concept 2: Retrieval Strategy

Use a two-stage strategy:
1. Candidate generation: hybrid keyword + semantic retrieval.
2. Candidate refinement: rerank top-N using stronger scorer.

Why hybrid?
- keyword retrieval catches exact identifiers,
- semantic retrieval catches meaning overlap.

---

## Concept 3: Prompt Construction with Grounding

Generation prompt should include only retrieved evidence.

Recommended structure:
- user question,
- concise evidence blocks with source IDs,
- explicit instruction: "Do not answer beyond provided evidence."

```python
prompt = f"""
You are a technical assistant.
Answer only using the evidence blocks below.
If evidence is insufficient, say so clearly.

Question: {question}

Evidence:
{evidence_blocks}

Output format:
- Answer
- Citations
- Confidence (low/medium/high)
"""
```

---

## Concept 4: Evaluation Hierarchy

Evaluate in this order:
1. Retrieval quality
2. Citation quality
3. Answer quality
4. Cost and latency

### Retrieval Metrics
- Recall@k
- MRR (optional)
- modality coverage (did retrieval include needed modality?)

### Answer Metrics
- faithfulness
- relevance
- completeness

### Operational Metrics
- p95 latency
- cost per query
- token usage per query

---

## Concept 5: Failure Taxonomy

Track failures by type:
- `R1`: relevant evidence missing from retrieval.
- `R2`: retrieved evidence present but badly ranked.
- `G1`: answer hallucinated despite correct evidence.
- `G2`: citations point to irrelevant source.
- `O1`: latency too high.

A failure log accelerates iteration and debugging.

---

## Suggested Build Plan

1. Build dataset index with modality tags.
2. Implement hybrid retrieval.
3. Add reranking and compare Recall@k.
4. Build grounded generation prompt.
5. Add citation parser/validator.
6. Run eval set and produce scorecard.
7. Optimize latency and token cost.
8. Write recommendations for next iteration.

---

## Reflection Questions

1. Which failure type appeared most often: retrieval or generation?
2. Did reranking improve Recall@k enough to justify latency?
3. Which modality contributed most to correct answers?
4. How often did the model say "insufficient evidence" appropriately?
5. What is your best next improvement for quality per dollar?

---

## Checklist

- [ ] I built a multimodal ingestion schema.
- [ ] I indexed text and non-text descriptors together.
- [ ] I implemented hybrid retrieval.
- [ ] I measured Recall@k on an eval set.
- [ ] I built grounded answer generation with citations.
- [ ] I evaluated faithfulness and citation correctness.
- [ ] I documented failure modes and mitigations.
- [ ] I reported latency/cost trade-offs.

---

*Guide for `phases/phase3/starters/STARTER_Weeks25-26_MultimodalRAG.ipynb` | Phase 3 | ML-AI-learning roadmap*
