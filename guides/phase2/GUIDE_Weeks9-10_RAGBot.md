# Guide: RAG Document Bot (Weeks 9-10)

## Beginner Start Here
This is your first retrieval-plus-LLM system. We build it step by step.

### What is RAG in simple words?
RAG has two parts:
1. Retrieve relevant text chunks from documents.
2. Generate an answer using only those chunks.

### Terms you must know first
- `Embedding`: text converted to a numeric vector.
- `Vector store`: database for similarity search on embeddings.
- `Chunking`: splitting large documents into smaller pieces.
- `Reranking`: reorder retrieved chunks for better quality.
- `Hallucination`: model says unsupported information.

### Modules used
- `langchain`: orchestration for retrieval and generation.
- `rank-bm25`: keyword retrieval.
- `sentence-transformers` (or API embeddings): vector representation.
- Vector DB client (`faiss`, `pinecone`, `weaviate`, etc.).

### How to study this guide
1. Build vector retrieval baseline.
2. Add keyword retrieval and fuse rankings.
3. Add reranker.
4. Run evals before saying it is done.

## Big Picture
Build a Retrieval-Augmented Generation (RAG) chatbot that can answer questions about custom documents.

**Why?** Move from generic LLM to domain-specific AI. RAG is foundation of enterprise AI applications.

**Key Skills:**
- Text embeddings and vector databases
- Document retrieval and semantic search
- LangChain for orchestration
- Prompt engineering for context
- Integration with OpenAI API

## 💼 Real-World Use Cases
- **Customer support bots:** Answer questions using product manuals, support articles, and FAQs.
- **Legal assistants:** Retrieve contract clauses and summarize obligations.
- **Sales enablement:** Provide reps with quick answers from internal playbooks and case studies.

---

## 📚 Recommended Documents for Weeks 9-10

RAG projects need **text documents**, not traditional datasets. Choose ONE option below:

### Option 1: Create Synthetic Documentation ✅ **EASIEST**
- **What:** Write your own documents about a topic
- **Format:** Plain text .txt files or markdown .md files
- **Example topics:**
  - Python programming guide (10-20 pages)
  - Product manual for imaginary SaaS app
  - Company handbook/policies
  - Tutorial series (e.g., "How to use FastAPI")
- **How to create:**
  ```python
  # Create sample documents
  documents = [
      "document1.txt: Contains X concepts...",
      "document2.txt: Contains Y features...",
      # etc.
  ]
  
  # Or use Lorem Ipsum to generate dummy text
  ```
- **Why:** Full control, no licensing issues, custom to your interests.
- **Effort:** 2-3 hours to write 2-3 good docs.

### Option 2: Wikipedia Articles 📖
- **What:** Free, public educational content
- **How to use:**
  ```python
  import wikipedia
  
  # Get article text
  page = wikipedia.page("Machine Learning")
  text = page.content
  
  # Save to file
  with open('ml_guide.txt', 'w') as f:
      f.write(text)
  ```
- **Topics:** Pick ANY Wikipedia article(s) (e.g., history, technology, science)
- **Why:** Free, diverse, public domain.
- **Limitation:** May not relate to specific domain.

### Option 3: Kaggle - Document Datasets 🗂️
- **What:** Pre-collected document corpora
- **Examples:**
  - https://www.kaggle.com/datasets/Cornell-University/arxiv (scientific papers)
  - https://www.kaggle.com/datasets/towardsdatascience/medium-articles (blog posts)
  - https://www.kaggle.com/datasets/liuhuiting/books-dataset (book content)
- **How to use:**
  ```python
  # Download from Kaggle
  # Extract text files
  documents = []
  for file in os.listdir('data/'):
      with open(f'data/{file}', 'r') as f:
          documents.append(f.read())
  ```
- **Why:** Curated, diverse, good quality.

### Option 4: GitHub README Files 💻
- **What:** Project documentation from open source
- **How to get:**
  ```python
  # Clone repositories
  ! git clone https://github.com/[user]/[repo].git
  
  # Extract README.md files
  import os
  docs = []
  for root, dirs, files in os.walk('.'):
      for file in files:
          if file == 'README.md':
              with open(os.path.join(root, file)) as f:
                  docs.append(f.read())
  ```
- **Popular repos:** Tensorflow, PyTorch, FastAPI (diverse technical content)
- **Why:** Real-world documentation, technical depth.

### Option 5: PDF Documents + Text Extraction 📄
- **What:** Convert PDF papers, manuals, reports to text
- **How to extract:**
  ```python
  from PyPDF2 import PdfReader
  
  reader = PdfReader("document.pdf")
  text = ""
  for page in reader.pages:
      text += page.extract_text()
  ```
- **Sources:**
  - Academic papers (ArXiv, Google Scholar)
  - Company whitepapers
  - News articles
  - Textbooks
- **Why:** Common in production RAG systems.
- **Setup:** Requires `pip install PyPDF2`

### Option 6: Your Own Content 🎯
- **What:** Use real documentation from a personal project
- **Examples:**
  - Your learning notes from this course
  - Blog posts you've written
  - Project documentation
  - Email chains (sanitized)
- **Why:** Most meaningful for learning, directly useful.

---

## 🚀 Quick Start Recommendation

**If you have 1 hour:** Use **Option 1** (write 3 simple docs about a topic you know)  
**If you have 30 min:** Use **Option 2** (grab Wikipedia articles)  
**If you want variety:** Use **Option 4** (GitHub READMEs - instant library of docs)

---

## Concept 1: The RAG Problem

**Traditional LLM:** No context beyond training data
```python
# Problem: LLM doesn't know about your specific docs
response = llm.ask("What's in document X?")
# Result: "I don't have access to that document"
```

**RAG Solution:** Give LLM your documents first
```python
# Better: Retrieve relevant excerpts, then ask
relevant_docs = retriever.search("document X question")
response = llm.ask(context=relevant_docs, question="...")
# Result: Answers based on your specific documents!
```

**Use Cases:**
- Customer support (answer from company docs)
- Legal analysis (search contracts/agreements)
- Research (search papers/studies)
- Knowledge management (internal documentation)

---

## Concept 2: Vector Embeddings

**What:** Convert text to numbers that capture meaning.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Text → Vector
text1 = "The cat sat on the mat"
embedding1 = model.encode(text1)  # [0.12, -0.45, 0.89, ...]

text2 = "A cat is sitting"
embedding2 = model.encode(text2)  # Similar to embedding1

text3 = "The weather is sunny"
embedding3 = model.encode(text3)  # Different from embeddings 1&2

# Similarity = distance between vectors
similarity = model.similarity(embedding1, embedding2)  # High (0.92)
similarity = model.similarity(embedding1, embedding3)  # Low (0.32)
```

**Key Insight:** Similar meanings = close vectors. This enables semantic search!

---

## Concept 3: Vector Store / Database

**What:** Storage system for embeddings + retrieval.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Create vector store
embeddings = OpenAIEmbeddings(api_key="...")
vectorstore = FAISS.from_documents(
    documents=[doc1, doc2, doc3, ...],
    embedding=embeddings
)

# Retrieve similar documents
query = "How do I reset my password?"
results = vectorstore.similarity_search(query, k=3)
# Returns top 3 most relevant documents
```

**Popular Vector Stores:**
- FAISS (local, fast, in-memory)
- Pinecone (cloud, scalable)
- Weaviate (open source, advanced)
- Chroma (lightweight, easy setup)

---

## Concept 4: Document Splitting

**What:** Break large documents into chunks for better retrieval.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Bad: One huge chunk = can't retrieve specific info
text = "Very long document... 100,000 words..."

# Good: Split into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Characters per chunk
    chunk_overlap=100  # Overlap to preserve context
)

chunks = splitter.split_text(text)
# chunks = ["First 1000 chars...", "Chars 900-1900...", ...]
```

**Why Overlap?** Information at boundaries stays intact.

---

## Concept 5: LangChain Orchestration

**What:** Framework connecting LLM, retriever, and tools.

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Setup
llm = ChatOpenAI(model_name="gpt-4", api_key="...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Combines all retrieved docs
    retriever=retriever
)

# Use
response = qa_chain.run("What's the refund policy?")
```

**Chain Types:**
- `stuff`: Combine all docs into one prompt
- `map_reduce`: Process docs separately, then combine
- `refine`: Build answer iteratively from each doc

---

## Concept 6: Prompt Engineering for RAG

**What:** Crafting prompts that leverage retrieved context.

```python
# Weak prompt (generic)
prompt = "Answer the question"

# Better prompt (with context instructions)
prompt = """
Use the following context to answer the question.
If you can't find the answer in the context, say "Not found in documents".

Context:
{context}

Question: {question}

Answer:
"""
```

**Key Elements:**
- Role definition ("You are a customer support agent")
- Context instructions ("Use provided documents")
- Format requirements ("Answer in JSON format")
- Fallback behavior ("If not found, say...")

---

## Concept 7: Handling Long Documents

**What:** Strategies when documents exceed token limits.

```python
# Problem: Context too long for LLM
context = get_all_relevant_docs()  # 50,000 tokens
llm_limit = 4096  # Can't fit!

# Solution 1: Return top-k chunks (truncate)
top_chunks = retriever.search(query, k=3)  # Only 3 chunks

# Solution 2: Summarize chunks before passing
summarized = [summarize(chunk) for chunk in chunks]
context = " ".join(summarized)  # Shorter context

# Solution 3: Use models with larger context
llm = ChatOpenAI(model_name="gpt-4-32k")  # 32k tokens!
```

---

## Concept 8: Conversation Memory

**What:** Keep chat history for multi-turn conversations.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# First turn
qa_chain.run(input="What's your return policy?")
memory.save_context({"input": "..."}, {"output": "..."})

# Second turn (remembers first)
qa_chain.run(input="How long does it take?")
# LLM now knows this is follow-up to refund question
```

**Memory Types:**
- BufferMemory: Keep all history (simple)
- SummaryMemory: Summarize old messages (save tokens)
- EntityMemory: Track key entities (advanced)

---

## Concept 9: Quality Metrics

**What:** Measuring RAG system effectiveness.

```python
# 1. Retrieval Quality
# Did we get relevant documents?
retrieved_docs = retriever.search(query, k=3)
relevance_score = measure_relevance(retrieved_docs, expected)

# 2. Answer Quality
# Did LLM generate good answer?
answer = qa_chain.run(query)
answer_quality = compare_with_gold_answer(answer, ground_truth)

# 3. Latency
# How fast is response?
import time
start = time.time()
response = qa_chain.run(query)
latency = time.time() - start
```

**Typical Metrics:**
- Precision@k: Relevant docs in top-k
- BLEU: Text similarity to gold answer
- Latency: Response time < 2 seconds

---

## Concept 10: Common Failure Modes

**What:** Issues to watch for.

```python
# Issue 1: Retrieving irrelevant documents
# Solution: Adjust similarity threshold
results = vectorstore.similarity_search_with_score(
    query, k=3, score_threshold=0.5
)

# Issue 2: Hallucinations (making up answers)
# Solution: Add to prompt "Say 'I don't know' if..."
prompt += "\nIf info not in documents, respond: 'I don't know'"

# Issue 3: Context length exceeded
# Solution: Check before passing
context = " ".join([d.page_content for d in docs])
if len(context) > MODEL_LIMIT:
    docs = docs[:k//2]  # Use fewer docs

# Issue 4: Slow retrieval with large vector store
# Solution: Use hierarchical indexing or chunking
```

---

## Challenge Approach

### Challenge 1-3: Document Preparation
- Collect sample documents (PDFs, text files)
- Split into chunks
- Create embeddings
- Store in vector database

### Challenge 4-6: Basic RAG
- Create retriever from vector store
- Build QA chain with LangChain
- Test with sample questions
- Iterate on prompt

### Challenge 7-9: Improvement & Optimization
- Add memory for multi-turn conversation
- Implement fallback handling
- Add streaming for long responses
- Optimize chunk sizes

### Challenge 10-12: Evaluation & Deployment
- Create test questions + gold answers
- Measure retrieval and answer quality
- Document system limitations
- Prepare for deployment

---

## Key Takeaways

✅ **RAG = Retrieval + Generation** (ground LLM in your documents)

✅ **Embeddings enable semantic search** (meaning-based, not keyword-based)

✅ **Chunking matters** (right chunk size = better retrieval)

✅ **LangChain simplifies orchestration** (connects components easily)

✅ **Prompt engineering is critical** (templating + instructions improve quality)

✅ **Test robustly** (retrieval quality, answer quality, edge cases)

---

## 🔍 Advanced RAG: Hybrid Search & Re-ranking ⭐

Pure vector search (semantic search) has a well-known failure mode: it struggles with **exact technical terms, part numbers, names, or abbreviations**. "GPT-4o" and "gpt-4o-mini" are very different models but semantically similar in embedding space.

The production fix is **Hybrid Search**: combine keyword matching (BM25) with vector similarity, then re-rank.

### Step 1 — BM25 (Keyword / Sparse Search)
BM25 is the classic search algorithm behind Elasticsearch and Google's early ranking. It scores documents by term frequency and inverse document frequency (TF-IDF family). It finds exact keyword matches that semantic search misses.

```python
# Install: pip install rank-bm25
from rank_bm25 import BM25Okapi

# Tokenize all documents
corpus = ["GPT-4o is a multimodal model", "gpt-4o-mini is smaller and cheaper", ...]
tokenized = [doc.lower().split() for doc in corpus]

bm25 = BM25Okapi(tokenized)

# Query
query = "gpt-4o-mini pricing"
bm25_scores = bm25.get_scores(query.lower().split())
# Returns raw BM25 score per document
```

### Step 2 — Fuse BM25 + Vector Scores (Reciprocal Rank Fusion)
Don't just average raw scores (they're on different scales). Use **Reciprocal Rank Fusion (RRF)**: convert each list to ranks, then combine.

```python
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple]:
    """
    rankings: list of ranked doc-id lists (one per retrieval method)
    k: constant preventing high weight on top-1 (usually 60)
    Returns: fused ranking as sorted (doc_id, score) list
    """
    scores = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Example usage
bm25_ranking   = sorted(range(len(corpus)), key=lambda i: bm25_scores[i], reverse=True)
vector_ranking = [idx for idx, _ in vector_search_results]  # from FAISS/Pinecone

fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
top10_ids = [doc_id for doc_id, score in fused[:10]]
```

### Step 3 — Cross-Encoder Re-ranking
After hybrid retrieval gives you top 10-20 candidates, a **Cross-Encoder** re-scores them against the query jointly (not separately). It's slower but much more accurate.

- **Bi-encoder** (SBERT): encodes query and document separately → fast, good recall at top-100
- **Cross-encoder**: encodes query+document together → slow, excellent precision, best for top-5

```python
# Install: pip install sentence-transformers
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Prepare (query, document) pairs for re-scoring
pairs = [(query, corpus[doc_id]) for doc_id in top10_ids]
scores = reranker.predict(pairs)

# Sort by re-ranked scores
reranked = sorted(zip(top10_ids, scores), key=lambda x: x[1], reverse=True)
final_top5 = [corpus[doc_id] for doc_id, score in reranked[:5]]
```

**Full pipeline summary:**
```
User Query
    ↓
BM25 → top-100 (keyword matches)          Vector DB → top-100 (semantic matches)
    ↓                                             ↓
         RRF Fusion → combined top-20
                    ↓
           Cross-Encoder re-ranking
                    ↓
              Final top-5 context
                    ↓
              LLM generates answer
```

---

## 📊 LLM Evaluation: RAGAS Framework ⭐

In traditional ML you have F1 and Accuracy. In LLMs, you need a systematic way to grade output quality — not just "vibes." **RAGAS** provides numerical scores for every RAG system component.

### Why this is critical
You cannot go to production without knowing: "Did my bot get better or worse after I changed the chunk size?" Without RAGAS, you're flying blind.

### The 4 RAGAS Metrics

| Metric | What it measures | Ideal |
|---|---|---|
| **Faithfulness** | Does the answer contain claims supported by the retrieved context? | 1.0 = no hallucination |
| **Answer Relevance** | Is the answer actually relevant to the question asked? | 1.0 = on-topic |
| **Context Precision** | Are the retrieved chunks actually useful for answering? | 1.0 = no noise |
| **Context Recall** | Was all necessary information retrieved? (needs ground truth) | 1.0 = nothing missed |

### Building Your Gold Dataset (evals/ folder)
A gold dataset is a set of question → expected answer pairs you create manually. This is your benchmark. Every time you change your RAG pipeline, you re-run against it.

```python
# evals/gold_dataset.json
gold_dataset = [
    {
        "question": "What is the refund policy for subscriptions?",
        "ground_truth": "Subscriptions can be refunded within 30 days of purchase.",
        "ground_truth_context": ["Our refund policy allows cancellation within 30 days..."]
    },
    # ... 20-50 more Q&A pairs
]
```

### Running RAGAS Evaluation

```python
# Install: pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Build evaluation dataset (from your RAG pipeline outputs)
eval_data = {
    "question":        ["What is the refund policy?", ...],
    "answer":          [rag_chain.invoke("What is the refund policy?"), ...],   # model output
    "contexts":        [[chunk1, chunk2], ...],   # retrieved chunks per question
    "ground_truth":    ["Refunds within 30 days", ...]   # from gold dataset
}

dataset = Dataset.from_dict(eval_data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result)
# Output:
# {'faithfulness': 0.87, 'answer_relevancy': 0.91, 'context_precision': 0.74, 'context_recall': 0.82}
```

### Interpreting Results
- **Faithfulness < 0.8:** model is hallucinating — improve your prompt, add explicit "only use retrieved context" instruction
- **Context Precision < 0.7:** retrieval is pulling irrelevant chunks — reduce chunk size, improve chunking strategy or add metadata filtering
- **Context Recall < 0.7:** retrieval is missing relevant information — increase top-K, switch to hybrid search
- **Answer Relevance < 0.8:** model is going off-topic — tighten the system prompt

Store results in `evals/ragas_results_YYYY-MM-DD.json` and track each score over time as you iterate.
