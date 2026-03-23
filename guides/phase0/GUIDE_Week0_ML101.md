# Guide: Week 0 — ML 101: What Is Machine Learning?
> **Phase 0 | Foundation** — Concepts before code.

---

## Beginner Start Here

This guide accompanies `STARTER_Week0_ML101.ipynb`. Read this **before** opening the notebook to build context, then keep it open as a reference while you work through the exercises.

### What This Guide Covers
- The big picture: AI vs ML vs Data Science
- Why machine learning works (the core idea)
- The 3 types of ML with detailed examples
- The 6-step ML workflow (your mental model for every project)
- Common misconceptions to avoid
- Where to go when you're stuck

### Key Terms
| Term | Plain English |
|------|---------------|
| **Artificial Intelligence (AI)** | The broad field of building intelligent machines |
| **Machine Learning (ML)** | AI systems that learn from data (a subset of AI) |
| **Deep Learning** | ML using large multilayer neural networks (a subset of ML) |
| **Data Science** | Extracting insights from data — uses ML as one tool |
| **Model** | A trained algorithm that maps inputs to predictions |
| **Training** | The process of fitting a model to data |
| **Feature** | An input variable — a column in your dataset |
| **Label / Target** | What you're predicting — the column the model must output |

---

## How to Study This Guide

1. **Read Section 1** (AI vs ML vs DS) and draw the nested diagram from memory
2. **Open the notebook** and run Section 1 code
3. **Read Section 2** (types of ML) and fill in the exercise before running the check cell
4. Continue reading each guide section then doing the matching notebook section
5. **Attempt the reflection questions** before running any cell that reveals answers

---

## Section 1: The Relationship Between AI, ML, and Data Science

### The Nested View
```
Artificial Intelligence (broadest)
  └── Machine Learning (learns from data)
        └── Deep Learning (very large neural nets)
```

**Data Science** is a separate (but overlapping) field focused on the full data lifecycle: collection → cleaning → analysis → modeling → communication. ML is one tool inside data science, but data science also includes statistics, visualization, and business communication.

### Why the Distinction Matters in Job Hunting
- **ML Engineer** job: building and deploying ML models at scale
- **Data Scientist** job: exploring data, running experiments, communicating insights
- **AI Engineer** job: building AI-powered products (RAG systems, agents, API integrations)
- This roadmap targets the **ML/AI Engineer** role trajectory

---

## Section 2: The 3 Types of Machine Learning

### Type 1: Supervised Learning
The model learns from labeled examples — `(input, correct output)` pairs.

**How it works:**
1. You have a dataset where each row is `(features, label)` — e.g., `([age, income, tenure], churned=1)`
2. The model finds a function `f(features) → label`
3. At prediction time, you give the model features and it returns a label

**Classification** — output is a discrete category:
- Spam/not spam
- Disease/no disease  
- Digit recognition (0-9)
- Sentiment (positive/negative/neutral)

**Regression** — output is a continuous number:
- House price ($)
- Customer lifetime value ($)
- Temperature forecast (°C)
- Stock price prediction

**Key algorithms you'll use:**
- Logistic Regression (classification — confusing name!)
- Linear Regression (regression)
- Decision Tree (both)
- Random Forest (both) — most used in industry
- XGBoost / LightGBM (both) — wins competitions
- Neural Networks (both) — most powerful, most data-hungry

### Type 2: Unsupervised Learning
No labels. The model finds hidden structure in data.

**Clustering** — group similar items:
- K-Means: partition customers into N behavioral groups
- DBSCAN: find dense regions of data, mark sparse regions as noise

**Dimensionality Reduction** — compress many features into fewer:
- PCA: find the directions of maximum variance
- UMAP/t-SNE: 2D/3D visualization of high-dimensional embeddings

**Anomaly Detection:**
- Isolation Forest: detect unusual transactions
- Autoencoder: reconstruct normal data, flag high reconstruction error as anomaly

### Type 3: Reinforcement Learning
An agent takes actions in an environment and learns from reward signals.

**Not covered until Phase 3**, but know the vocabulary:
- **Agent**: the learner (e.g., game-playing AI, RLHF-tuned LLM)
- **Environment**: what the agent acts in
- **Reward**: scalar feedback (+1 for good action, -1 for bad)
- **Policy**: the agent's strategy for choosing actions

---

## Section 3: The 6-Step ML Workflow

This is the most important thing to internalize in Phase 0. Every single ML project follows this cycle. Memorize it.

### Step 1: Define the Problem
**Questions to answer before writing any code:**
- What are we predicting? (classification or regression?)
- What data do we have?
- How will success be measured? (accuracy? revenue impact? latency?)
- What are the constraints? (real-time? privacy? interpretability?)
- What is the cost of a wrong prediction?

**Bad example:** "Let's build a churn model."  
**Good example:** "We want to predict 30-day churn probability for active customers with 3+ months tenure, using billing and usage data available in our data warehouse, with recall ≥ 70% at precision ≥ 50%."

### Step 2: Collect & Explore Data (EDA)
- Load data: `pd.read_csv()`, database query, API call
- Inspect: `.head()`, `.info()`, `.describe()`
- Distributions: histograms, box plots
- Relationships: scatter plots, correlation heatmap
- Missing values: `df.isnull().sum()`
- Class balance: `df['target'].value_counts(normalize=True)`

**Key insight from EDA informs everything downstream.** Don't skip it.

### Step 3: Prepare the Data
- **Handle missing values**: fill with median/mode/model, or drop
- **Encode categorical features**: `LabelEncoder`, `OneHotEncoder`, target encoding
- **Scale numeric features**: `StandardScaler` (mean=0, std=1) or `MinMaxScaler` (range 0-1)
- **Feature engineering**: create new columns from existing ones (total_charges = monthly × tenure)
- **Train/test split**: `train_test_split(X, y, test_size=0.2, random_state=42)`

### Step 4: Train a Model
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

Start simple. Logistic Regression as baseline → Random Forest → XGBoost → tune the best.

### Step 5: Evaluate
**Classification metrics:**
- Accuracy: `(TP + TN) / total` — misleading with class imbalance
- Precision: `TP / (TP + FP)` — how many predicted positives are actually positive?
- Recall: `TP / (TP + FN)` — how many actual positives did the model find?
- F1: harmonic mean of precision and recall
- ROC-AUC: overall ranking ability across all thresholds

**Regression metrics:**
- MAE: mean absolute error
- RMSE: root mean squared error (penalizes large errors more)
- R²: proportion of variance explained

### Step 6: Deploy & Monitor
- Serialize the model: `joblib.dump(model, 'model.pkl')`
- Serve via REST API: FastAPI + Pydantic
- Monitor: data drift (input distribution changes), model drift (performance degrades)
- Retrain on a schedule or when drift is detected

---

## Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "More data always helps" | Only if the data is relevant. Noisy data hurts. |
| "Deep learning is always best" | For tabular data, XGBoost often beats neural nets |
| "High accuracy = good model" | A model that predicts 99% of emails as "not spam" has 99% accuracy but is useless |
| "The model is a black box I can't understand" | Many techniques exist: SHAP, LIME, feature importance |
| "Once deployed, the model is done" | Models degrade — monitoring and retraining are essential |

---

## When You Get Stuck

1. **Re-read the error message** — Python errors tell you exactly what went wrong and where
2. **Check the documentation**: numpy.org/doc, pandas.pydata.org, scikit-learn.org
3. **Search the error on Stack Overflow** — most errors have been asked before
4. **Print intermediate values** — add `print(variable)` to see what's going wrong
5. **Simplify** — make a small example that reproduces the problem

---

## Resources

- [sklearn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Kaggle Learn: Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning)

---

*Guide for `STARTER_Week0_ML101.ipynb` | Phase 0 | ML-AI-learning roadmap*
