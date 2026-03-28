# 📋 Quick Reference: ML/Data Science Cheat Sheet

Keep this handy while you code!

## Quick Navigation

Use this file in this order when building projects:
1. [Tool stack and minimum practical usage](#part-1-tooling-overview)
2. [Model and metric selection guides](#part-2-decision-guides)
3. [Core data work: pandas, NumPy, visualization](#part-3-core-data-work)
4. [Classical ML workflow, models, and evaluation](#part-4-classical-ml-workflow)
5. [Deep learning, RAG, serving, and Docker](#part-5-deep-learning-llms-and-serving)
6. [Common patterns, debugging, and stuck-state recovery](#part-6-operations-and-troubleshooting)

## Part 1: Tooling Overview

---

## Popular Python ML/AI Engineer Stack (2026)

This section lists the most common tools used by ML/AI engineers across companies.

### 1) Core Data and Math (use every week)
- `numpy`: fast numerical arrays and matrix math.
- `pandas`: tabular data loading, cleaning, and analysis.
- `matplotlib` and `seaborn`: charts for EDA and model diagnostics.

### 2) Classical Machine Learning (must know)
- `scikit-learn`: preprocessing, baseline models, metrics, and pipelines.
- `xgboost`, `lightgbm`, `catboost`: high-performing gradient-boosted tree models.

### 3) Deep Learning (one framework deeply, know both names)
- `torch` (PyTorch): dominant in research and many production teams.
- `tensorflow`/`keras`: still common in enterprise and legacy stacks.

### 4) NLP and LLM App Stack (must know for GenAI roles)
- `transformers`: model loading and inference (Hugging Face ecosystem).
- `tokenizers`: efficient text tokenization.
- `peft`: parameter-efficient fine-tuning (LoRA/adapters).
- `sentence-transformers`: embedding models for retrieval.

### 5) Retrieval and Vector Search
- `faiss`: fast local vector similarity search.
- `qdrant-client`, `weaviate-client`, `pinecone-client`: production vector DB options.
- `rank-bm25`: keyword retrieval for hybrid search.

### 6) Agent and Orchestration Layer
- `langchain`: chains, tools, memory, and agent workflows.
- `llama-index`: retrieval/indexing abstraction for document systems.

### 7) API and Serving Layer
- `fastapi`: model and agent serving APIs.
- `pydantic`: request/response schema validation.
- `uvicorn`: ASGI server for FastAPI apps.

### 8) Experiment Tracking and Evaluation
- `mlflow`, `wandb`: experiment tracking and model lineage.
- `ragas`: RAG quality metrics.
- `evidently`: data drift and quality monitoring.

### 9) Data Quality and Pipelines
- `great-expectations`: data validation checks.
- `prefect` or `airflow`: workflow orchestration.
- `dbt`: analytics transformations (SQL-first teams).

### 10) Performance and Scale (learn after foundations)
- `polars`, `duckdb`: fast local analytics.
- `pyspark`: distributed data processing.
- `ray`: distributed Python execution for training/inference workloads.

### Suggested Learning Priority
1. `numpy`, `pandas`, `matplotlib`, `seaborn`
2. `scikit-learn` and baseline model evaluation
3. `torch` and `transformers`
4. `faiss` + one vector DB + `rank-bm25`
5. `fastapi` + `pydantic`
6. `mlflow`/`wandb` + `ragas` + `evidently`

### Interview Signal Stack (high ROI)
If you can ship projects with this combination, you are highly competitive:
- `pandas` + `scikit-learn`
- `torch` + `transformers`
- `faiss` + `rank-bm25` + reranker
- `fastapi` + `pydantic`
- `mlflow`/`wandb` + eval dashboard

## How To Use These Tools (Practical)

This section answers: "When do I use this package, and what does a minimum real example look like?"

### 1) `pandas` and `numpy` (start of every project)
Use when: loading, cleaning, and transforming raw data.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/customers.csv')
df['tenure_months'] = df['tenure_days'] / 30
df['is_high_value'] = (df['monthly_spend'] > 100).astype(int)

# Result: clean feature table for modeling
X = df[['tenure_months', 'monthly_spend', 'is_high_value']]
y = df['churn']
```

### 2) `scikit-learn` (baseline model quickly)
Use when: building first model and measuring baseline metrics.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print('F1:', f1_score(y_test, pred))
```

### 3) `xgboost` / `lightgbm` / `catboost` (strong tabular model)
Use when: baseline works but you need better accuracy on structured/tabular data.

```python
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

### 4) `torch` (deep learning training)
Use when: working with images, text transformers, or neural networks.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

x = torch.randn(8, 10)
logits = model(x)
print(logits.shape)  # [8, 2]
```

### 5) `transformers` (LLM and NLP models)
Use when: running sentiment, summarization, QA, or text generation.

```python
from transformers import pipeline

summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
text = 'Long article text goes here...'
summary = summarizer(text, max_length=80, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
```

### 6) `sentence-transformers` + `faiss` (semantic retrieval)
Use when: building RAG search over documents.

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

docs = ['reset password guide', 'pricing and billing', 'deployment steps']
embedder = SentenceTransformer('all-MiniLM-L6-v2')
emb = embedder.encode(docs).astype('float32')

index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

q = embedder.encode(['how to change password']).astype('float32')
dist, idx = index.search(q, k=2)
print([docs[i] for i in idx[0]])
```

### 7) `rank-bm25` (keyword retrieval for hybrid search)
Use when: vector search misses exact keywords/IDs/part numbers.

```python
from rank_bm25 import BM25Okapi

tokenized_docs = [d.split() for d in docs]
bm25 = BM25Okapi(tokenized_docs)
scores = bm25.get_scores('password reset'.split())
print(scores)
```

### 8) `langchain` (agent and RAG orchestration)
Use when: connecting retriever + prompt + model + tools in one flow.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
prompt = ChatPromptTemplate.from_template('Answer clearly: {q}')
chain = prompt | llm
result = chain.invoke({'q': 'What is recall in ML?'})
print(result.content)
```

### 9) `fastapi` + `pydantic` + `uvicorn` (serve model in production)
Use when: exposing your model to web/mobile/backend apps.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    x1: float
    x2: float

@app.post('/predict')
def predict(req: PredictRequest):
    score = 0.7 * req.x1 + 0.3 * req.x2
    return {'score': score}

# run: uvicorn main:app --reload
```

### 10) `mlflow` / `wandb` / `ragas` / `evidently` (prove quality)
Use when: tracking experiments, evaluating model quality, and monitoring drift.

```python
# MLflow example
import mlflow

with mlflow.start_run():
    mlflow.log_param('model', 'xgboost')
    mlflow.log_metric('f1', 0.84)
```

```python
# RAGAS is used to score RAG outputs (faithfulness, relevance, context quality)
# Store these scores each iteration to prove your system is improving.
```

### 11) `great-expectations` / `prefect` / `airflow` (data reliability)
Use when: moving from notebooks to scheduled, reliable pipelines.

```python
# Example concept:
# - Great Expectations validates data schema
# - Prefect schedules pipeline jobs
# - Airflow orchestrates DAG workflows
```

### Practical Rule
- Start simple: pandas + sklearn baseline.
- Add complexity only when a metric or requirement demands it.
- Every added tool must answer: "What problem does this solve right now?"

---

## Part 2: Decision Guides

## 🧠 Explaining Model Selection

This section answers: "How do I choose the right model instead of guessing?"

### Start With These Questions
1. Is this classification, regression, clustering, ranking, generation, or retrieval?
2. Is the data tabular, text, image, audio, or time series?
3. Do I need interpretability, or is raw performance more important?
4. Is the dataset small, medium, or large?
5. What kind of mistakes are most expensive?
6. Do I need fast inference or real-time prediction?

### Default Model Choice by Problem Type

#### Tabular Classification
Start with:
- Logistic Regression for a simple baseline
- Random Forest for a strong non-linear baseline
- XGBoost/LightGBM/CatBoost when you need stronger performance

Use Logistic Regression when:
- You want a fast baseline
- You need coefficients and simpler explanation
- The dataset is not extremely complex

Use tree-based models when:
- Relationships are non-linear
- Feature interactions matter
- You have mixed numeric/categorical patterns

Rule of thumb:
- Baseline: Logistic Regression
- Stronger next step: Random Forest
- High-performance tabular work: XGBoost/LightGBM/CatBoost

#### Tabular Regression
Start with:
- Linear Regression or Ridge
- Random Forest Regressor
- Gradient boosting if simple linear models underperform

Use linear models when:
- You want a simple benchmark
- Interpretability matters
- The signal is roughly additive or linear

Use tree/boosting regressors when:
- Effects are non-linear
- There are many interactions
- You care more about predictive performance than coefficient interpretation

#### Text Classification / NLP
Start with:
- TF-IDF + Logistic Regression for a cheap, strong baseline
- Transformer models when task complexity is higher

Use TF-IDF + linear model when:
- You need a quick text baseline
- Dataset is moderate and labels are clean
- Compute budget is limited

Use transformers when:
- Context, semantics, and nuance matter
- The task depends on meaning more than keywords
- You can afford heavier inference/training cost

#### Image Tasks
Start with:
- Transfer learning using pretrained CNN or ViT models

Do not start by training from scratch unless:
- You have a very large dataset
- You have a strong reason to avoid pretrained models

#### Time Series / Forecasting
Start with:
- Naive baseline (last value, rolling mean)
- Tree-based models with lag features for practical forecasting
- Specialized forecasting models only when needed

Key rule:
- Respect time order. Never random-shuffle time series in a way that leaks the future.

#### Clustering / Unsupervised Work
Start with:
- KMeans for simple segmentation
- DBSCAN when cluster shape/noise matters
- PCA/UMAP for dimensionality reduction before visualization

### Selection by Constraint

#### If you need interpretability
Prefer:
- Logistic Regression
- Linear Regression / Ridge
- Small Decision Tree

Why:
- Easier to explain feature influence
- Better for audits and business communication

#### If you need strongest tabular performance
Prefer:
- XGBoost
- LightGBM
- CatBoost

Why:
- These are often top performers on structured data
- They capture non-linear patterns and interactions well

#### If you need fast training and baseline iteration
Prefer:
- Logistic Regression
- Linear models
- Small Random Forest

Why:
- Faster feedback loop
- Easier to debug pipeline issues before tuning advanced models

#### If you have very little data
Prefer:
- Simpler models first
- Strong regularization
- Careful cross-validation

Why:
- Complex models can overfit small datasets quickly

#### If you have lots of text, image, or audio data
Prefer:
- Pretrained deep learning models
- Fine-tuning rather than training from scratch

### Fast Selection Cheat Sheet

| Situation | Example | Good first model | Good next model |
|---|---|---|---|
| Binary classification, tabular | Customer churn, fraud yes/no, loan default | Logistic Regression | Random Forest / XGBoost |
| Multi-class tabular | Support ticket category, product type prediction | Logistic Regression / Random Forest | XGBoost / CatBoost |
| Numeric prediction | House prices, sales amount, delivery time | Linear Regression / Ridge | Random Forest Regressor / XGBoost |
| Text classification | Spam detection, sentiment analysis, ticket routing | TF-IDF + Logistic Regression | Transformers |
| Image classification | Product photos, defect detection, animal species | Pretrained CNN/ViT | Fine-tuned stronger pretrained model |
| Retrieval / RAG | Help-center search, internal docs assistant, policy lookup | BM25 baseline + embeddings | Hybrid retrieval + reranker |

### What Usually Goes Wrong
- Picking a complex model before building a baseline
- Choosing by popularity instead of problem type
- Using accuracy only on imbalanced classification
- Comparing models without keeping preprocessing consistent
- Tuning models before checking data quality and leakage

### A Good Practical Workflow
1. Build the simplest valid baseline.
2. Evaluate against the right metric.
3. Identify the failure mode.
4. Pick the next model based on that failure mode.
5. Change one major thing at a time.

Example:
- If Logistic Regression underfits tabular data, try Random Forest.
- If Random Forest is better but still misses complex interactions, try XGBoost.
- If text keywords work but meaning is missed, move from TF-IDF to transformers.

### Decision Rule
- Simple data + need explanation: start linear.
- Structured/tabular data + need strong performance: use boosted trees.
- Unstructured text/image/audio: start with pretrained deep models.
- Small data or early exploration: use the simplest model that gives a valid baseline.

### Example Scenarios

#### Scenario 1: Customer churn prediction from CSV data
- Data type: tabular
- Problem: binary classification
- Good first model: Logistic Regression
- Good next model: Random Forest or XGBoost
- Why: churn data is structured, and important patterns often come from interactions across tenure, contract type, and price

#### Scenario 2: House price prediction
- Data type: tabular
- Problem: regression
- Good first model: Linear Regression or Ridge
- Good next model: Random Forest Regressor or XGBoost Regressor
- Why: linear models create a clear benchmark, while tree models usually capture non-linear effects better

#### Scenario 3: Spam detection from email text
- Data type: text
- Problem: classification
- Good first model: TF-IDF + Logistic Regression
- Good next model: Transformer classifier
- Why: keywords often give a strong baseline, but transformers help when meaning and phrasing matter more

#### Scenario 4: Product image classification
- Data type: images
- Problem: classification
- Good first model: pretrained vision model
- Good next model: stronger fine-tuned pretrained model
- Why: transfer learning is usually far more efficient than training from scratch

#### Scenario 5: Support chatbot retrieval
- Data type: text documents
- Problem: retrieval/ranking
- Good first system: BM25 + embedding search baseline
- Good next system: hybrid retrieval + reranker
- Why: exact keyword match and semantic similarity both matter in real support systems

---

## 📏 Explaining Metric Selection

This section answers: "How do I choose the metric that actually matches the goal?"

### Start With These Questions
1. What mistake is most expensive?
2. Are classes balanced or imbalanced?
3. Is the output a yes/no decision, a ranking, or a numeric value?
4. Will people act directly on the prediction?
5. Do large errors need extra penalty?

### Classification Metrics

#### Accuracy
Use when:
- Classes are reasonably balanced
- False positives and false negatives have similar cost

Avoid when:
- The positive class is rare

Example:
- A churn dataset with 95% non-churn can produce high accuracy with a useless model

#### Precision
Use when:
- False positives are expensive
- You want high-confidence positive predictions

Example scenarios:
- Fraud alerts that trigger manual review
- Sales outreach where bad leads waste team time

#### Recall
Use when:
- False negatives are expensive
- Missing positives is worse than reviewing extra cases

Example scenarios:
- Churn prevention
- Disease screening
- Safety issue detection

#### F1 Score
Use when:
- Precision and recall both matter
- The dataset is imbalanced

Example scenario:
- Spam filtering where both missed spam and blocked real email matter

#### ROC-AUC
Use when:
- You want to compare ranking quality across thresholds
- The class split is not extremely skewed

#### PR-AUC
Use when:
- The positive class is rare
- You care about the quality of positive-case retrieval

### Regression Metrics

#### MAE
Use when:
- You want average error in original units
- Large misses should not dominate the score too much

Example:
- Delivery time prediction in minutes

#### RMSE
Use when:
- Large errors should be penalized more heavily
- Big misses are especially painful

Example:
- Demand forecasting where large misses create inventory problems

#### R-squared
Use when:
- You want a high-level measure of explained variance

Do not use alone:
- It can look good while actual prediction error is still too large to be useful

### Retrieval / RAG Metrics

Use:
- Recall@k when missing relevant documents is costly
- Precision@k when showing irrelevant documents is costly
- Faithfulness/groundedness when answer correctness depends on retrieved evidence

### Fast Metric Selection Cheat Sheet

| Situation | Primary metric | Why |
|---|---|---|
| Balanced classification | Accuracy or F1 | Reasonable when classes are not heavily skewed |
| Imbalanced classification | Recall, Precision, F1, PR-AUC | Accuracy can be misleading |
| Churn prediction | Recall or F1 | Missing likely churners is costly |
| Fraud detection | Precision and Recall | Both misses and false alarms matter |
| Forecasting numeric value | MAE or RMSE | Use original-unit error or stronger penalty for large misses |
| Retrieval / RAG | Recall@k, Precision@k, groundedness | Need correct retrieval and supported answers |

### Example Metric Scenarios

#### Scenario 1: Churn model
- Business goal: catch at-risk customers
- Best starting metric: recall
- Supporting metric: precision
- Why: missing churners is usually worse than reviewing too many customers

#### Scenario 2: Fraud alerts
- Business goal: detect fraud without overwhelming investigators
- Best starting metric: precision and recall together
- Supporting metric: PR-AUC
- Why: both missed fraud and too many false alerts are expensive

#### Scenario 3: House price prediction
- Business goal: estimate sale price accurately
- Best starting metric: MAE
- Supporting metric: RMSE
- Why: MAE is easy to explain in dollars, while RMSE highlights very large misses

#### Scenario 4: Search over help-center docs
- Business goal: retrieve the right documents in top results
- Best starting metric: Recall@k
- Supporting metric: Precision@k
- Why: first make sure the correct document appears, then improve ranking quality

### Metric Selection Mistakes
- Using accuracy for highly imbalanced problems
- Reporting only one metric when tradeoffs matter
- Ignoring threshold tuning for probabilistic classifiers
- Comparing metrics across different datasets or different splits
- Optimizing a metric that does not match the real business action

### Decision Rule
- If missing positives is worst, optimize recall.
- If false alarms are worst, optimize precision.
- If both matter, use F1 or track precision and recall together.
- If outputs are ranked, include AUC or ranking metrics.
- If prediction is numeric, start with MAE and add RMSE when large misses matter.

---

## 🧭 Problem-Type Mini Recipes

Use these when you want a fast default workflow for a common project type.

### 1) Tabular Classification Recipe
- Examples: churn, fraud, loan default, spam labels in a CSV
- Start with: train/test split, preprocessing pipeline, Logistic Regression baseline
- Next model: Random Forest or XGBoost
- Primary metrics: recall, precision, F1, PR-AUC depending on class balance and error cost
- Common risks: leakage, imbalance, inconsistent category handling

Recommended flow:
1. Clean target and fix data types.
2. Split before fitting preprocessing.
3. Build a pipeline with imputation and encoding.
4. Train Logistic Regression baseline.
5. Compare with Random Forest or XGBoost.
6. Tune threshold if business action depends on recall/precision tradeoff.

### 2) Tabular Regression Recipe
- Examples: house prices, sales forecasting, delivery time prediction
- Start with: Linear Regression or Ridge
- Next model: Random Forest Regressor or XGBoost Regressor
- Primary metrics: MAE first, RMSE second
- Common risks: skewed targets, leakage from future information, outliers dominating error

Recommended flow:
1. Check target distribution and outliers.
2. Build a preprocessing pipeline.
3. Train linear baseline.
4. Compare with non-linear regressor.
5. Review large-error cases, not only average score.

### 3) Text Classification Recipe
- Examples: sentiment, spam, support ticket routing
- Start with: TF-IDF + Logistic Regression
- Next model: transformer classifier
- Primary metrics: F1, recall, precision depending on the operational goal
- Common risks: label noise, class imbalance, text leakage from metadata

Recommended flow:
1. Clean obvious null and duplicate text records.
2. Build TF-IDF baseline.
3. Review error examples by reading actual text.
4. Move to transformer only if baseline misses semantic meaning.

### 4) Retrieval / RAG Recipe
- Examples: internal docs assistant, support search, knowledge-grounded chatbot
- Start with: BM25 baseline plus embedding search
- Next system: hybrid retrieval with reranker
- Primary metrics: Recall@k, Precision@k, groundedness/faithfulness
- Common risks: poor chunking, weak metadata filters, hallucination without source support

Recommended flow:
1. Start with a small gold query set.
2. Test keyword and semantic retrieval separately.
3. Combine them into hybrid retrieval.
4. Add reranking only after retrieval coverage is acceptable.
5. Evaluate answers against sources, not style alone.

### 5) Image Classification Recipe
- Examples: product photos, defect detection, animal species classification
- Start with: pretrained CNN or ViT
- Next model: fine-tuned stronger pretrained backbone
- Primary metrics: accuracy for balanced datasets, recall/precision/F1 when class costs differ
- Common risks: too little data, label quality issues, train/validation leakage through near-duplicate images

Recommended flow:
1. Inspect class balance and label quality.
2. Use transfer learning first.
3. Add augmentation carefully.
4. Review confusion matrix by class.
5. Investigate failure images, not just scores.

---

## Part 3: Core Data Work

## 🐼 Pandas Essentials

### Loading & Saving
```python
import pandas as pd

df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx')
```

### Exploration
```python
df.head()                    # First 5 rows
df.tail(10)                  # Last 10 rows
df.shape                     # (rows, columns)
df.dtypes                    # Data types
df.info()                    # Full summary
df.describe()                # Statistics
df.columns                   # Column names
```

### Cleaning
```python
# Missing values
df.isnull().sum()            # Count NaN
df.dropna()                  # Remove NaN rows
df.fillna(0)                 # Fill with 0
df.fillna(df.mean())         # Fill with mean

# Duplicates
df.duplicated().sum()        # Count duplicates
df.drop_duplicates()         # Remove duplicates

# Data types
df['col'] = df['col'].astype(float)
```

### Transforming and Fixing Data Types (Very Common)
```python
# 1) Inspect object columns before conversion
obj_cols = df.select_dtypes(include=['object']).columns
for c in obj_cols:
    print(c, 'unique:', df[c].nunique(dropna=False))

# 2) Clean raw strings first (strip spaces, normalize case)
df['city'] = (
    df['city']
    .astype('string')
    .str.strip()
    .str.lower()
)

# 3) Convert numeric-like strings safely
# Bad values become NaN instead of crashing
df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')

# 4) Parse datetime safely
df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce', utc=True)

# 5) Convert low-cardinality text columns to category
# Great for memory + clearer semantics
cat_candidates = ['contract_type', 'payment_method', 'internet_service']
for c in cat_candidates:
    if c in df.columns:
        df[c] = df[c].astype('category')

# 6) Optional: set an explicit category order when business logic needs it
if 'risk_level' in df.columns:
    df['risk_level'] = pd.Categorical(
        df['risk_level'],
        categories=['low', 'medium', 'high'],
        ordered=True
    )

# 7) Fix common placeholders that should be missing values
df = df.replace({'': pd.NA, 'NA': pd.NA, 'N/A': pd.NA, 'unknown': pd.NA})

# 8) Basic consistency check after transformations
print(df.dtypes)
print(df[['monthly_charges', 'signup_date']].head())
```

When to use `category`:
- Good for columns with repeated labels and relatively low unique values (plan type, region, status).
- Avoid converting free-text columns (comments, descriptions) to category.
- For ML: use encoding (like OneHotEncoder) in a pipeline rather than manual integer coding.

Quick pattern for training prep:
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

num_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(include=['category', 'object', 'string']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])
```

### Selection & Filtering
```python
df['col_name']               # Select column
df[['col1', 'col2']]         # Multiple columns
df.loc[0]                    # Row by label
df.iloc[0]                   # Row by position
df[df['age'] > 18]           # Filter
df[(df['age'] > 18) & (df['salary'] > 50000)]  # Multiple filters
```

### Grouping & Aggregation
```python
df.groupby('category').sum()
df.groupby('category').mean()
df.groupby(['cat1', 'cat2']).agg({'salary': 'mean', 'age': 'max'})
df['col'].value_counts()     # Count occurrences
df['col'].nunique()          # Unique values
```

---

## 🔢 NumPy Essentials

```python
import numpy as np

# Arrays
arr = np.array([1, 2, 3])
arr = np.zeros(10)           # Array of zeros
arr = np.ones(10)            # Array of ones
arr = np.arange(0, 10, 2)    # 0, 2, 4, 6, 8
arr = np.linspace(0, 10, 5)  # 5 equal spaces 0-10

# Operations
arr + 5                      # Add to all
arr * 2                      # Multiply all
np.sqrt(arr)                 # Square root
np.mean(arr)                 # Average
np.std(arr)                  # Standard deviation
np.max(arr)                  # Maximum
```

---

## 📊 Matplotlib & Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot
plt.plot(x, y)
plt.show()

# Histogram
plt.hist(data, bins=20)

# Scatter plot
plt.scatter(x, y)

# Box plot
plt.boxplot([data1, data2])

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x)
axes[0, 1].hist(y)

# Seaborn
sns.heatmap(correlation_matrix, annot=True)
sns.boxplot(data=df, x='category', y='value')
sns.scatterplot(data=df, x='x', y='y', hue='category')
```

---

## Part 4: Classical ML Workflow

## 🤖 Scikit-learn Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Separate numeric and categorical columns
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()

# 3. Build preprocessing + model in one pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols),
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# 4. Make predictions
pred = model.predict(X_test)

# 5. Evaluate
print(classification_report(y_test, pred))
```

Best default pattern:
- Use a `Pipeline` so preprocessing is identical in train, test, and inference.
- Split before fitting transformations.
- Start with Logistic Regression for a baseline, then replace only the final estimator when comparing models.

---

## 🎯 Classification Models

### Logistic Regression (Simple)
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
```

### Decision Tree (Interpretable)
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)
```

### Random Forest (Often Best)
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### XGBoost (Advanced)
```python
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 📈 Regression Models

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### Ridge Regression (Prevents overfitting)
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

### Random Forest Regression
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 📊 Evaluation Metrics

### Classification
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)          # Overall correctness
precision = precision_score(y_test, y_pred)        # True positives / predicted positives
recall = recall_score(y_test, y_pred)              # True positives / actual positives
f1 = f1_score(y_test, y_pred)                      # Harmonic mean of precision & recall

cm = confusion_matrix(y_test, y_pred)              # True/False positives/negatives
print(classification_report(y_test, y_pred))
```

### Regression
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)           # Average squared error
rmse = np.sqrt(mse)                                # Root mean squared error
mae = mean_absolute_error(y_test, y_pred)         # Average absolute error
r2 = r2_score(y_test, y_pred)                      # Explained variance (0-1)
```

---

## 🔄 Cross-Validation & Hyperparameter Tuning

Use [HYPERPARAMETER_TUNING_GUIDE.md](/Users/jjcatulle/Desktop/ML-AI-learning/docs/HYPERPARAMETER_TUNING_GUIDE.md) for the full explanation-first version.

### What tuning actually means
- Tuning means changing model or system settings on purpose, then comparing results with a validation process.
- You tune after you have a valid baseline, not before.
- You do not tune on the test set repeatedly.
- You should tune the settings most connected to the current failure mode.

Examples:
- Churn model missing positives: tune threshold or `class_weight` first.
- House price model making large misses: tune regularization or tree depth.
- RAG bot missing the right source: tune chunking and retrieval before prompt wording.

### Cross-Validation
Use when:
- one random split may be too noisy
- dataset is not huge
- you want a more stable comparison between settings

Why it helps:
- it reduces luck from one split
- it gives a more honest average score across folds

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
```

### Grid Search
Use when:
- you have a small, clear parameter space
- you want a simple and reproducible search

Watch out for:
- too many combinations
- searching before you know what behavior you are trying to improve

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Good tuning workflow
1. Build a baseline.
2. Protect the test set.
3. Choose the metric that matches the goal.
4. Pick 1 to 3 meaningful parameters.
5. Compare against baseline, not just another tuned run.
6. Record what improved and what got worse.

---

## Part 5: Deep Learning, LLMs, and Serving

## 🧠 Deep Learning (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Setup
model = NeuralNet(input_size=10, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
```

---

## 🤗 LangChain Essentials

```python
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Create model
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Explain {topic} in simple terms."
)

# Create chain
chain = prompt | llm

# Run
result = chain.invoke({"topic": "machine learning"})
print(result.content)
```

### RAG (Retrieval-Augmented Generation)
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA

# 1. Load documents
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Store in vector DB
vector_store = Pinecone.from_documents(chunks, embeddings, index_name="my-index")

# 5. Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# 6. Query
answer = qa_chain.invoke({"query": "What is the document about?"})
```

---

## 🐳 FastAPI Model Serving

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="ML Model API")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define request/response schemas
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

# Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    features = [request.feature1, request.feature2, request.feature3]
    prediction = model.predict([features])[0]
    confidence = 0.95
    return PredictionResponse(prediction=prediction, confidence=confidence)

@app.get("/health")
async def health():
    return {"status": "ok"}

# Run with: uvicorn main:app --reload
```

---

## 🐳 Docker Quick Start

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Building & Running
```bash
# Build image
docker build -t my-ml-app .

# Run container
docker run -p 8000:8000 my-ml-app

# Run with volume mount (for development)
docker run -v $(pwd):/app -p 8000:8000 my-ml-app
```

---

## Part 6: Operations and Troubleshooting

## 📌 Common Patterns

### Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Scale Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Handle Missing Values
```python
df = df.dropna()  # Remove rows with NaN
# OR
df = df.fillna(df.mean())  # Fill with mean
```

### Encode Categorical Variables
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(df[['category']])
```

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
```

---

## ⏱️ Debugging Tips

### Check for Overfitting
- High training accuracy, low test accuracy
- Solution: Use regularization, more data, simpler model

### Check for Underfitting
- Low both training and test accuracy
- Solution: More features, more complex model, more training

### Class Imbalance
- Minority class predictions are bad
- Solution: Use `class_weight='balanced'` or stratified split

### Scaling Issues
- Model performance is bad
- Solution: Scale features with StandardScaler or MinMaxScaler

### Convergence Issues
- Model takes too long to train
- Solution: Lower learning rate, smaller batch size, normalize data

---

## 🎓 When Stuck

1. **Read the error message carefully** - usually tells you exactly what's wrong
2. **Google the error** + library name
3. **Check official documentation**
4. **Try a simpler version** of your code
5. **Ask on Stack Overflow** with reproducible example

---

## 📚 Keep This File Open While Coding!

Print it, bookmark it, or keep it in your IDE. Reference it constantly!

---

*Last Updated: March 18, 2026*
