# Weeks 33-39: NLP, Semantic Search, LLM Ops (Deep Dive)

## 1. Core NLP
- Tokenization, embedding spaces, transformer architecture.
- Sequence classification, named entity recognition, question answering.

## 2. Semantic Search
- Sentence embeddings (SBERT, OpenAI text-embedding-3-small, TensorFlow Hub).
- Vector stores: FAISS, Milvus, Pinecone, Weaviate.
- Retrieval pipelines: k-NN, reranking, hybrid retrieval.

## 3. LLM Applications
- Prompt engineering patterns: zero-shot, few-shot, chain-of-thought.
- Safety filters, temperature/top-p control, instruction tuning.
- Fine-tuning and adapters with `peft`/`transformers`.

## 4. Production Considerations
- Cost monitoring, token budgeting, cache and rate limits.
- Latency optimization and multi-model fallback.
- Logging and evaluating hallucinations.

## 5. Reference
- Papers: "Attention Is All You Need", "BERT", "RAG".
- Hugging Face course and OpenAI docs.

## 6. Challenge
- Build semantic retrieval with embeddings + RAG on a custom knowledge base; compare recall@k.
- Add prompt-chain QA with reference citation.
