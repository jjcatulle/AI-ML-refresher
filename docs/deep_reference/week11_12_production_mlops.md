# Weeks 11-12: Production ML and MLOps (Deep Dive)

## 1. ML Flow in Prod
- Data pipelines: ingestion, validation, transformation, feature store.
- Model pipeline: train/test, packaging, deployment.
- Monitoring data and model quality as code (CI/CD) with GitHub Actions.

## 2. Serve Model
- API design with FastAPI (predict, health check, metadata endpoints).
- Containerize with Docker + multi-stage build + minimal base image.
- Use model versioning: tags, artifacts, and `mlflow` model registry.

## 3. Observability
- Metrics: latency, throughput, error rate.
- Data drift and prediction drift detection via `river` or `alibi-detect`.
- Logging with structured logs to ELK/Prometheus.

## 4. Resilience
- Blue-green/canary deployment.
- Retraining triggers and automated model-promote pipeline.
- Explainable AI checks for compliance.

## 5. Reference
- Book: "Building Machine Learning Powered Applications".
- Blog posts: "MLOps with FastAPI and Docker".
- Tools: `prefect` / `airflow`, `dvc`, `kubeflow`.

## 6. Vector DB Indexing Strategies ⭐
Searching 1 million vectors is easy. Searching 1 million vectors **while filtering for `user_id=123` and `date > 2024`** efficiently is a hard engineering problem. This section covers exactly that.

### What is a Vector Index?
A vector index is a data structure that allows fast approximate nearest-neighbor (ANN) search. Without it, searching for the most similar vector requires comparing against every single stored vector — O(n) per query.

### HNSW (Hierarchical Navigable Small World) — Default for Most Use Cases
**How it works:** Builds a multi-layer graph. Higher layers are sparse long-range connections; lower layers are dense short-range. Search starts at the top layer and navigates downward.

**Performance:**
- Query speed: Very fast even at scale (sub-millisecond for millions of vectors)
- Recall: Very high (>95% with default settings)
- Memory: High — entire graph stored in RAM
- Build time: Slower than IVF

**Best for:** Real-time queries, smaller-medium datasets (fits in memory), when recall matters most

```python
import faiss
import numpy as np

d = 384          # embedding dimension (e.g., SBERT output)
M = 32           # number of connections per node (higher = better recall, more memory)

# Build HNSW index
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efSearch = 64   # search expansion factor (higher = better recall, slower)

# Add vectors
vectors = np.random.randn(100_000, d).astype('float32')
index.add(vectors)

# Query: find 5 nearest neighbors
query = np.random.randn(1, d).astype('float32')
distances, indices = index.search(query, k=5)
print('Nearest indices:', indices)
```

### IVF (Inverted File Index) — For Large Scale
**How it works:** Clusters vectors into `nlist` Voronoi cells at index build time. At query time, only `nprobe` cells are searched instead of all of them.

**Performance:**
- Query speed: Fast, especially when `nprobe` is low
- Recall: Lower than HNSW (trades recall for speed)
- Memory: Lower — quantized vectors, centroids only
- Build time: Faster than HNSW

**Best for:** 10M+ vectors, when memory is constrained, batch queries

```python
nlist = 100   # number of Voronoi cells
nprobe = 10   # how many cells to search at query time (higher = better recall)
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

# Must train IVF (learns cluster centroids)
index_ivf.train(vectors)
index_ivf.add(vectors)
index_ivf.nprobe = nprobe

distances, indices = index_ivf.search(query, k=5)
```

### Metadata Filtering — The Hard Problem
Pure ANN search returns the K most similar vectors globally. In real applications, you need to filter **before or after** similarity search:

- "Find similar documents **only from user_id=123**"
- "Find similar products **only in category='shoes' and price < 100**"

**Strategy 1: Post-filtering (common but flawed)**
```python
# Retrieve top-K*10, then filter by metadata
results = index.search(query, k=500)           # over-fetch
filtered = [r for r in results if r.user_id == 123][:5]  # filter down
# Problem: if few results match the filter, you get poor recall
```

**Strategy 2: Pre-filtering with purpose-built vector DBs**
Databases like Qdrant, Weaviate, and Pinecone support native filtered ANN:
```python
# Qdrant example with metadata filter
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(":memory:")
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="user_id",  match=MatchValue(value=123)),
            FieldCondition(key="category", match=MatchValue(value="tech")),
        ]
    ),
    limit=5
)
```

### Quick Comparison Table
| Feature | HNSW | IVF |
|---|---|---|
| Recall | Very high | Tunable |
| Query speed | Fast | Faster at scale |
| Memory | High (RAM) | Lower |
| Build time | Slow | Fast |
| Best scale | <10M vectors | 10M+ vectors |
| Use it when | Real-time, high recall | Scale, memory-constrained |

## 7. Challenge
- Create a GitHub workflow to train and push a model artifact, then run unit + integration tests.
- Build a FAISS HNSW index on 100K synthetic vectors. Benchmark query time vs brute-force (`IndexFlatL2`).
- Add metadata filtering using Qdrant in-memory: filter by two fields and compare recall vs unfiltered search.
