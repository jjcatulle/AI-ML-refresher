# Guide: Capstone Project (Weeks 33-39)

## Beginner Start Here
This final project is large, but we break it into small, manageable steps.

### What this capstone proves
You can design and ship a reliable AI system, not just train a model.

### Terms you must know first
- `Architecture`: blueprint of system components and data flow.
- `Pipeline`: ordered processing steps.
- `Evals`: repeatable tests for quality and safety.
- `Guardrails`: checks that block unsafe or sensitive outputs.
- `Governance`: fairness, privacy, and compliance rules.

### How to work through capstone
1. Build smallest end-to-end version first.
2. Add critic and eval checks.
3. Add guardrails and monitoring.
4. Improve one metric at a time.

## Big Picture
Build an end-to-end production AI system: problem → multimodal context → agentic orchestration → evals → guardrails → deployment.

**Why?** In 2026, being job-ready means architecting reliable autonomous systems, not only training a model.

**Key Skills:**
- Problem formulation and scoping
- Data collection and wrangling
- Model experimentation and selection
- Evaluation and business metrics
- Multi-agent orchestration (planner/researcher/writer/critic)
- LLM-as-a-Judge and adversarial evaluation design
- Guardrails for privacy, safety, and prompt-injection resilience
- Deployment and monitoring
- Documentation and communication

## 💼 Real-World Use Cases
- **Product analytics:** Build end-to-end A/B testing and model deployment pipelines.
- **Fintech:** Build credit scoring systems with risk monitoring and compliance.
- **Healthcare:** Develop prediction systems that integrate with operational dashboards and alerting.

---

## 🎯 Recommended Capstone Project Ideas

### Option 0: Production-Grade Autonomous Agent ⭐ RECOMMENDED
- **Challenge:** Build an autonomous agent that ingests:
  - one 50-page PDF manual,
  - one 5-minute tutorial video (transcript + key frames),
  - one CSV of sales data,
  and generates a quarterly performance report.
- **Must Include:**
  - Multi-agent flow: Planner → Researcher → Writer → Critic
  - Critic node that fact-checks against retrieved evidence
  - Guardrail layer that redacts customer PII before output
  - Auto-eval script scoring report faithfulness and completeness
- **Why:** Directly maps to enterprise AI architect roles.

---

Choose ONE project below that excites you. This will be your production ML system:

### Option 1: Customer Churn Predictor (Extended) ✅ **RECOMMENDED**
- **Dataset:** Telecom/SaaS customer churn data (from Week 4-5)
- **Scope:** Add to previous project with monitoring + retraining
- **Deliverables:**
  - Better model (ensembles, hyperparameter tuning)
  - FastAPI deployment
  - Data drift monitoring
  - Automated retraining pipeline
  - Business dashboard
- **Why:** Build on existing knowledge, realistic end-to-end project.
- **Datasets:** Kaggle Telco, IBM HR, Bank Customer Churn

### Option 2: Recommendation System 🎬
- **What:** Predict products/movies/content users will like
- **Dataset sources:**
  - MovieLens: https://grouplens.org/datasets/movielens/
  - Amazon Reviews: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
  - Book Ratings: https://www.kaggle.com/datasets/sootersaalu/amazon-top-50-bestselling-books-2009-2019
- **Concepts:**
  - Collaborative filtering
  - Content-based filtering
  - Matrix factorization
  - A/B testing
- **Deliverables:**
  - Recommendation API
  - Personalization engine
  - Performance metrics
- **Why:** Fun, real-world application, user-facing.

### Option 3: Image Classification System 📷
- **What:** Classify images into categories (e.g., medical scans, product types)
- **Dataset sources:**
  - Kaggle Plant Disease: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  - Kaggle Furniture: Search "furniture classification"
  - Medical imaging: https://www.kaggle.com/datasets/c1d9c2c1d9c2/covid-19-radiography-database
- **Concepts:**
  - Transfer learning with ResNet/VGG
  - Data augmentation
  - Model optimization (quantization, pruning)
  - Real-time inference
- **Deliverables:**
  - Web UI for uploading images
  - API for classification
  - Performance monitoring
- **Why:** Deep learning capstone, impressive results.

### Option 4: Time Series Forecasting 📈
- **What:** Predict future values (stock prices, website traffic, energy consumption)
- **Dataset sources:**
  - Stock data: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
  - Website traffic: https://www.kaggle.com/datasets/bolbol/wikipedia-daily-views
  - Energy: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices
- **Concepts:**
  - ARIMA, SARIMA
  - LSTM networks
  - Seasonality detection
  - Uncertainty quantification
- **Deliverables:**
  - Forecast API
  - Confidence intervals
  - Alert system (if forecast exceeds threshold)
- **Why:** Different problem type (temporal), useful for planning.

### Option 5: Natural Language Processing (Sentiment Analysis) 💬
- **What:** Analyze sentiment of reviews/social media
- **Dataset sources:**
  - Movie reviews: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  - Twitter sentiment: https://www.kaggle.com/datasets/kazanova/sentiment140
  - Product reviews: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- **Concepts:**
  - Text preprocessing (tokenization, stemming)
  - TF-IDF, word embeddings
  - BERT/RoBERTa fine-tuning
  - Multi-class classification
- **Deliverables:**
  - Sentiment API
  - Dashboard showing sentiment trends
  - Topic extraction
- **Why:** NLP skills, useful for brand monitoring.

### Option 6: Fraud Detection System 🚨
- **What:** Identify fraudulent transactions/activity
- **Dataset sources:**
  - Credit card fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  - Transaction fraud: https://www.kaggle.com/datasets/kelvinkelvinkelvin/credit-card-fraud-prediction
- **Concepts:**
  - Class imbalance handling (SMOTE, class weights)
  - Anomaly detection
  - Real-time scoring
  - Explainability (SHAP values)
- **Deliverables:**
  - Risk scoring API
  - Alert system for high-risk transactions
  - Explainability dashboard (why flagged?)
- **Why:** Business-critical, security awareness.

### Option 7: Healthcare/Medical Diagnosis 🏥
- **What:** Predict disease risk or diagnose from medical data
- **Dataset sources:**
  - Diabetes: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
  - Heart disease: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
  - Cancer: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Concepts:**
  - Medical data handling (privacy, ethics)
  - Model interpretability (doctors need to understand)
  - Handling missing data
  - Cross-validation for reliability
- **Deliverables:**
  - Risk prediction tool
  - Model explainability
  - Clinical use guidelines
- **Why:** High-impact, introduces ethics/fairness considerations.

### Option 8: Your Own Problem 🎯
- **What:** Choose a real-world problem you care about
- **Guidelines:**
  - Find public dataset (or create synthetic data)
  - Clear target variable
  - 1000+ rows ideally
  - Business value of solving it
- **Examples:**
  - Predict apartment rental prices (personal finance)
  - Classify Spotify playlists by mood (music taste)
  - Forecast coffee sales by weather (business interest)
- **Why:** Most meaningful for your career and portfolio.

---

## 📋 Capstone Checklist

Whichever project you choose, ensure it includes:

- [ ] **Problem definition:** Why does this matter?
- [ ] **Data collected:** Cleaned, explored, understood
- [ ] **Multiple models tried:** Compare approaches
- [ ] **Strong evaluation:** Metrics, cross-validation, test results
- [ ] **Deployment ready:** API or web interface
- [ ] **Monitoring setup:** Data drift, model performance tracking
- [ ] **Agentic orchestration:** At least 3 specialized agents with handoff protocol
- [ ] **Evaluation harness:** RAGAS + LLM-as-a-Judge on a gold eval set
- [ ] **Adversarial tests:** Prompt injections, jailbreak attempts, edge-case failures
- [ ] **Guardrails:** PII masking + harmful output checks before user response
- [ ] **Shadow mode:** New version runs in background and logs side-by-side outcomes
- [ ] **Documentation:** README, architecture diagram, how to run
- [ ] **Business summary:** ROI, insights, recommendations
- [ ] **Portfolio ready:** Code on GitHub, demo-able to employers

### Job-Ready Upgrade Mapping
| Phase | Baseline Project | 2026 Upgrade |
|---|---|---|
| 2 | Simple Chatbot | Multi-Agent Researcher (3 agents collaborating) |
| 2 | RAG Bot | Multimodal RAG (text + image/video retrieval) |
| 4 | Model API | Self-healing pipeline (retries + drift-triggered fallback) |
| 4 | Portfolio Assembly | Governance-first system (safety and bias audit included) |

---

## 🚀 Quick Decision Guide

**Choose by interest:**
| If you like... | Choose... |
|---|---|
| Structured data | Churn, Fraud, Healthcare |
| Images | Image Classification |
| Videos/Sequences | Time Series |
| Text | NLP Sentiment |
| Recommendations | Recommendation System |
| Your own idea | Custom Project |

**Choose by difficulty:**
| Difficulty | Project |
|---|---|
| Medium | Churn, Sentiment Analysis |
| Medium-Hard | Fraud Detection, Healthcare |
| Hard | Image Classification, Recommendations |
| Hard+ | Time Series (LSTM), Your custom idea |

---

## Concept 1: Capstone Components

**What:** Full ML pipeline from conception to production.

```
1. Problem Definition
   ↓
2. Data Collection
   ↓
3. EDA & Preprocessing
   ↓
4. Feature Engineering
   ↓
5. Model Development
   ↓
6. Evaluation
   ↓
7. Deployment
   ↓
8. Monitoring & Iteration
```

---

## Concept 2: Problem Formulation

**What:** Turning business questions into ML problems.

```
Poor: "Use ML to improve our business"

Better: "Predict which customers will churn in next 30 days 
         with >80% accuracy so we can target retention efforts"

Even Better: "Build system to:
- Predict churn probability for each customer
- Rank customers by churn risk
- Integration with email campaign system
- Update daily with new data
- Track ROI: cost of campaign vs saved customer value"
```

**Questions to Answer:**
1. What are we predicting? (target variable)
2. Why does it matter? (business value)
3. How will it be used? (deployment)
4. What's good enough? (success metric)
5. What constraints? (latency, cost, fairness)

---

## Concept 3: Data Strategy

**What:** Systematic data handling.

```python
# Data Quality Checklist
✓ Data dictionary: What is each feature?
✓ Missing values: How much? Imputation strategy?
✓ Outliers: Are they real or errors?
✓ Class imbalance: Balanced dataset?
✓ Temporal validity: Is old data still relevant?
✓ Privacy: Contains PII? GDPR compliant?

# Data Split Strategy
- Train: 70% (learn patterns)
- Validation: 15% (tune hyperparameters)
- Test: 15% (final evaluation, touch ONCE)

# Time-series consideration (if applicable)
- Use chronological split, not random!
- Train: months 1-24
- Validation: months 25-36  
- Test: months 37-48
```

---

## Concept 4: Feature Engineering

**What:** Creating/transforming features strategically.

```python
# Feature categories

# Raw features (as-is)
age, income, location

# Engineered features (created)
age_squared = age**2
age_group = "senior" if age > 65 else "adult"
months_as_customer = (today - signup_date).days / 30

# Aggregated features (from related data)
avg_purchase_amount = customer_purchases.mean()
days_since_last_purchase = (today - last_purchase).days

# Domain features (expert knowledge)
customer_lifetime_value = revenue_from_customer
seasons_active = len(unique_purchase_months)

# Derived features (combinations)
purchase_frequency = num_purchases / months_as_customer
```

---

## Concept 5: Model Selection & Experimentation

**What:** Systematic model comparison.

```python
# Baseline
- Always predict most common class (accuracy ceiling)
- Or predict mean value (regression)

# Simple models (fast, interpretable)
- Logistic Regression
- Decision Tree
- KNN

# Ensemble models (better performance)
- Random Forest
- Gradient Boosting
- XGBoost

# Complex models (if needed)
- Neural Networks
- LSTMs (time-series)
- CNNs (images)

# Comparison matrix
Model               Accuracy  Precision  F1    Training Time  Inference Time
Logistic Reg        0.72      0.68       0.70  5 min          <1 ms
Random Forest       0.78      0.75       0.77  10 min         <10 ms
XGBoost             0.81      0.79       0.80  15 min         <5 ms
Neural Network      0.80      0.78       0.79  60 min         <50 ms

→ Choose XGBoost (high accuracy, reasonable time)
```

---

## Concept 6: Evaluation Strategy

**What:** Measuring success beyond accuracy.

```python
# Performance metrics
- Primary: Designed for your problem (F1 for imbalanced, AUC for ranking)
- Secondary: Interpretability, stability, fairness

# Business metrics
- Precision: False positives cost us money
- Recall: False negatives lose us customers
- Latency: Response time < 100ms required
- Cost: Model inference cost < $0.01 per call

# Real-world validation
# Does top-ranked predict work in practice?
predictions = model.predict(new_data)
actual_results = check_reality_after_2_weeks()
correlation = measure_correlation(predictions, actual_results)
```

---

## Concept 7: Deployment Pipeline

**What:** Moving model from dev to production.

```
Model Development (Notebook)
↓
Packaging (Flask/FastAPI app)
↓
Testing (unit + integration)
↓
Containerization (Docker)
↓
Staging (test in prod-like env)
↓
Production (serve to users)
↓
Monitoring (track performance)
```

---

## Concept 8: Monitoring & Maintenance

**What:** Model performance in production.

```python
# Data drift detection
# Distribution of features changes over time
# → Model performance degrades
# → Retrain on recent data

# Model drift detection
# Model predictions diverge from reality
# → Business assumptions changed
# → Need investigation

# Monitoring checks
Daily:
- Prediction distribution (many NaNs? All same class?)
- Latency (>100ms? Server issue?)
- Error rate (<0.1%? Degradation?)

Weekly:
- Accuracy vs holdout test set
- Feature importance changes
- Business metric impact

Monthly:
- Model vs baseline
- Data/model drift analysis
- Retrain decision
```

---

## Concept 9: Documentation

**What:** Making project understandable to others.

```
1. README.md
   - Problem statement
   - Quick start
   - Results summary

2. ARCHITECTURE.md
   - System diagram
   - Components and their interactions
   - Data flow

3. MODEL.md
   - EDA findings
   - Real features used
   - Model selection rationale
   - Performance metrics
   - Limitations

4. DEPLOYMENT.md
   - Deployment steps
   - Environment setup
   - Monitoring
   - Rollback procedure

5. NOTEBOOK
   - Reproducible analysis
   - Well-commented code
   - Explains key decisions
```

---

## Concept 10: Project Scope Management

**What:** Keeping project manageable.

```
Week 1-2: Problem definition + data collection
Week 3-4: EDA + cleaning
Week 5-6: Feature engineering + baseline models
Week 7: Model selection + hyperparameter tuning
Week 8: Evaluation + error analysis
Week 9-13: Deployment + monitoring setup
Final: Documentation + presentation

Red flags:
- Spending >1 week perfecting score (diminishing returns)
- Using complex model without trying simple one first
- No baseline model to beat
- No test set held aside
- No monitoring plan
```

---

## Challenge Approach

Choose one of:
1. **Recommendation System** - Predict what user wants
2. **Fraud Detection** - Detect anomalies
3. **Sentiment Analysis** - Classify text opinions
4. **Time Series Forecasting** - Predict future values
5. **Computer Vision** - Classify images

### Week 1-2: Problem & Data
- Define problem clearly
- Find/collect 10k+ samples
- Create data dictionary

### Week 3-4: Exploration
- EDA with 10+ visualizations
- Check quality, outliers, patterns
- Create baseline model

### Week 5-6: Development
- Engineer 20+ features
- Try 5+ models
- Document selection rationale

### Week 7: Tuning & Evaluation  
- Optimize best model
- Cross-validation
- Error analysis

### Week 8: Deployment
- Create API (FastAPI)
- Containerize (Docker)
- Document thoroughly

### Final: Presentation
- 1-page summary
- Key visualizations
- Business impact
- Limitations

---

## Key Takeaways

✅ **Define problem clearly** (determines everything else)

✅ **Baseline model first** (know what to beat)

✅ **Real-world concerns dominate** (accuracy < deployment + monitoring)

✅ **Documentation matters** (future you won't remember decisions)

✅ **Production ≠ Notebooks** (error handling, monitoring, updates)

✅ **Iterate based on feedback** (perfect first try unlikely)
