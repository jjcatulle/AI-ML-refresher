# Guide: RAG Document Bot (Weeks 9-10)

## Big Picture
Build a Retrieval-Augmented Generation (RAG) chatbot that can answer questions about custom documents.

**Why?** Move from generic LLM to domain-specific AI. RAG is foundation of enterprise AI applications.

**Key Skills:**
- Text embeddings and vector databases
- Document retrieval and semantic search
- LangChain for orchestration
- Prompt engineering for context
- Integration with OpenAI API

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
