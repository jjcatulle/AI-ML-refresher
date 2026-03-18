# Guide: Capstone Project (Weeks 33-39)

## Big Picture
Build an end-to-end production ML system: problem → data → model → deployment.

**Why?** Real-world ML is much more than training. This project integrates ALL skills learned.

**Key Skills:**
- Problem formulation and scoping
- Data collection and wrangling
- Model experimentation and selection
- Evaluation and business metrics
- Deployment and monitoring
- Documentation and communication

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
