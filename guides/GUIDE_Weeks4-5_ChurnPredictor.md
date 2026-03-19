# Weeks 4-5: Customer Churn Predictor - Learning Guide

Before tackling the 12 challenges, understand what you're building!

## 💼 Real-World Use Cases
- **Telecom:** Predict which customers will cancel subscriptions and target retention offers.
- **SaaS:** Identify trial users likely to churn and personalize onboarding.
- **Finance:** Detect when clients are likely to close their accounts.

---

## 📊 Recommended Datasets for Weeks 4-5

Choose ONE dataset below to build your churn predictor:

### Option 1: Kaggle - Telco Customer Churn ✅ **BEST FOR THIS PROJECT**
- **What:** Telecom company customer churn data
- **Size:** 7,043 customers, 19 features (demographics, services, billing)
- **Target:** Churn (Yes/No)
- **Where:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **How to load:**
  ```python
  df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
  ```
- **Why:** Perfect fit for this project, real business data, clear churn target.
- **Data quality:** Clean, ~11% churn rate (realistic imbalance).

### Option 2: Kaggle - IBM HR Analytics Attrition 👥
- **What:** Employee attrition and satisfaction data
- **Size:** 1,470 employees, 34 features (salary, department, role, satisfaction scores)
- **Target:** Attrition (Yes/No)
- **Where:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- **How to load:**
  ```python
  df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
  ```
- **Why:** Similar structure to churn, teach you features beyond just billing.
- **Interesting twist:** Built-in from IBM Watson Analytics.

### Option 3: Kaggle - Bank Customer Churn 🏦
- **What:** Bank customer churn with credit scores, age, salary, etc.
- **Size:** 10,000 customers, 10 features
- **Target:** Exited (0 or 1)
- **Where:** https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers
- **How to load:**
  ```python
  df = pd.read_csv('Churn_Modelling.csv')
  ```
- **Why:** Financial institution data, interesting demographic patterns.
- **Data quality:** Balanced target (20% churn).

### Option 4: UCI ML - Customer Churn 📞
- **What:** Another telecom churn dataset with slightly different features
- **Size:** 3,333 customers, 11 features
- **Where:** https://archive.ics.uci.edu/dataset/693/telephone+service+churn
- **How to load:**
  ```python
  df = pd.read_csv('churn.data', header=None)
  df.columns = ['feature1', 'feature2', ...]  # Rename as needed
  ```
- **Why:** Smaller, simpler, good for debugging before tackling larger datasets.

---

## 🎯 Big Picture: What is a Complete ML Project?

Every ML project follows this pipeline:

```
Real-World Problem
    ↓
Define Success Metrics
    ↓
Collect Data
    ↓
Explore & Understand Data (EDA)
    ↓
Clean & Prepare Data
    ↓
Train Model(s)
    ↓
Evaluate Performance
    ↓
Interpret Results
    ↓
Deploy to Production
```

**Your Churn Predictor = ALL of these steps!**

---

## 📚 Concept 1: The Churn Problem

### Business Context
A telecom company loses money when customers leave (churn). They want to:
1. Predict WHO will leave
2. Understand WHY they leave
3. Take action BEFORE they leave

### Example
- If we know John will churn, offer him a discount
- If we know churn is driven by high prices, lower them across the board

### Your Job
Build a model that predicts churn, identify drivers, and communicate findings.

---

## 📚 Concept 2: Problem Definition

### Target Variable
**Churn**: Binary outcome (0 or 1)
- 1 = Customer left (churned)
- 0 = Customer stayed

### Features (Inputs)
What predicts churn?
- **tenure**: How long they've been a customer (months)
  - Insight: New customers more likely to leave
- **monthly_charges**: Their bill amount
  - Insight: High bills → more likely to leave
- **contract_month_to_month**: Contract type
  - Insight: Month-to-month = easy to leave
- **tech_support**: Did they get support?
  - Insight: With support → less likely to leave

### Success Metrics
How will we measure if the model is good?

| Metric | What It Means | Ideal | Formula |
|--------|---|---|---|
| **Accuracy** | % of correct predictions | >80% | (TP+TN)/(Total) |
| **Precision** | Of predicted churns, how many actually churned? | High | TP/(TP+FP) |
| **Recall** | Of actual churns, how many did we catch? | High | TP/(TP+FN) |
| **F1** | Balance between Precision & Recall | High | 2×(Prec×Rec)/(Prec+Rec) |

**For churn**: We care about **Recall** most (catch the churners!)

---

## 📚 Concept 3: Data Preparation

### Step 1: Load Data
```python
# Create or load dataset
df = ...  # 5000-70000 rows, 8+ columns
```

### Step 2: Explore
Already know this from Week 1:
- `.shape`, `.describe()`, `.value_counts()`, `.corr()`

### Step 3: Separate Features & Target
```python
X = df.drop('churn', axis=1)  # Features (inputs)
y = df['churn']               # Target (output)
```

### Step 4: Scale Features
**Why scale?**
- Some features are 0-100, others are 0-1
- Unscaled data confuses some algorithms
- Scaling brings all to 0-1 range

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit = learn mean/std, transform = apply
```

### Step 5: Train/Test Split
**Why split?**
- Train on 80%, test on 20%
- Prevents overfitting (memorizing answers)
- Tests real-world performance

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # Reproducibility
    stratify=y               # Keep class balance
)
```

---

## 📚 Concept 4: Training Models

### What's a Model?
A mathematical function that learns from data:

```
Input (features) → Model → Output (prediction)
Example: [tenure=60, monthly_charges=75, ...] → Model → 0.05 (churn probability)
```

### Two Models to Try

#### Model 1: Logistic Regression
**Pros**: Simple, interpretable, fast  
**Cons**: Doesn't capture complex relationships

```python
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(random_state=42, max_iter=1000)
model1.fit(X_train, y_train)  # Learn from training data
```

#### Model 2: Random Forest
**Pros**: Often more accurate, handles complexity  
**Cons**: Harder to interpret

```python
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
```

### How Training Works
1. Model starts with random guesses
2. Compares predictions to actual values
3. Calculates error
4. Adjusts itself to reduce error
5. Repeats until good

---

## 📚 Concept 5: Making Predictions

### Two Types of Predictions

#### Prediction Type 1: Class (0 or 1)
```python
y_pred = model.predict(X_test)  # Returns 0 or 1
# Result: [0, 1, 0, 0, 1, ...]  <- Direct predictions
```

#### Prediction Type 2: Probability (0.0-1.0)
```python
y_pred_proba = model.predict_proba(X_test)  # Returns probabilities
# Result: [[0.95, 0.05], [0.20, 0.80], ...]
# First column = probability of 0 (no churn)
# Second column = probability of 1 (churn)
```

**Why both?**
- **Class**: "Will customer churn? Yes/No"
- **Probability**: "What's the confidence? 95% sure, 80% sure, etc."

Use probability for ROC curves and scoring.

---

## 📚 Concept 6: Evaluation Metrics Explained

### The Confusion Matrix
Compare predicted vs actual:

```
              Predicted 0   Predicted 1
Actual 0    |    TN         FP       |  Negatives
Actual 1    |    FN         TP       |  Positives
            
Legend:
TN = True Negative (correctly said "no churn")
FP = False Positive (wrongly said "churn")
FN = False Negative (missed a churn!)
TP = True Positive (correctly said "churn")
```

### Metrics Explained

**Accuracy** = How often correct?
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Example: Got 800/1000 correct = 80% accurate
Problem: Ignores class imbalance
```

**Precision** = Of predicted churns, how many were right?
```
Precision = TP / (TP + FP)
Example: Predicted 100 churns, 80 were correct = 80% precision
Focus on: Don't waste money on wrong predictions
```

**Recall** = Of actual churns, how many did we catch?
```
Recall = TP / (TP + FN)
Example: 100 actual churns, caught 80 = 80% recall
Focus on: Don't miss customers we could save
```

**F1-Score** = Balance of Precision & Recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Use when: Class imbalance matters
```

**ROC-AUC** = Overall model quality
```
Values: 0.5 (random guess) to 1.0 (perfect)
Plot trade-off between True Positive Rate & False Positive Rate
Perfect: Curve reaches top-left corner
```

---

## 📚 Concept 7: Feature Importance

### What's Feature Importance?
Which features matter most for prediction?

```
If model cares about:
- Tenure: 40%
- Monthly Charges: 30%
- Contract Type: 20%
- Age: 10%

Then Tenure is most important!
```

### How to Extract (Random Forest)
```python
importances = model.feature_importances_
# Result: [0.40, 0.30, 0.20, 0.10]

# Create DataFrame for clarity
import pandas as pd
feature_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

feature_df.head()  # Top 5 features
```

### Why This Matters
- **Business**: Focus on top drivers (e.g., "Lower prices to reduce churn")
- **Model**: Verify features make sense (e.g., "Tenure matters" = expected)

---

## 📚 Concept 8: ROC Curves

### What's an ROC Curve?
Shows trade-off between catching churners (True Positive Rate) and false alarms (False Positive Rate).

### Reading ROC Curves
- **Diagonal line** = Random guess (50/50)
- **Curve in top-left** = Good model
- **Curve in top-right** = Bad model
- **AUC** = Area under curve (0.5=random, 1.0=perfect)

### How to Create
```python
from sklearn.metrics import roc_curve, roc_auc_score

# Get probabilities (not class predictions!)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plot
plt.plot(fpr, tpr, label=f'Model (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

## 📚 Concept 9: Creating a Prediction Function

### Why Create a Function?
Instead of using the whole pipeline every time, create reusable code:

```python
def predict_churn(tenure, monthly_charges, total_charges, age, num_services, 
                   contract_month_to_month, internet_service, tech_support):
    """
    Predict if a customer will churn.
    
    Inputs:
    - tenure: Months as customer
    - monthly_charges: Monthly bill
    - ... other features
    
    Returns:
    - Dictionary with prediction, probability, risk level
    """
    
    # 1. Create feature vector
    features = [[tenure, monthly_charges, ...]]  # List of lists
    
    # 2. Scale (use the SAME scaler from training!)
    features_scaled = scaler.transform(features)
    
    # 3. Predict
    probability = model.predict_proba(features_scaled)[0, 1]
    prediction = model.predict(features_scaled)[0]
    
    # 4. Categorize risk
    if probability > 0.6:
        risk = 'High'
    elif probability > 0.4:
        risk = 'Medium'
    else:
        risk = 'Low'
    
    return {
        'churn_probability': probability,
        'churn_prediction': 'Yes' if prediction == 1 else 'No',
        'risk_level': risk
    }

# Test it
result = predict_churn(60, 75, 5000, 45, 5, 0, 1, 1)
print(result)
```

---

## 📚 Concept 10: Business Communication

### Translating Technical Results → Business Actions

| Technical Result | Business Translation | Action |
|---|---|---|
| Recall = 85% | "We catch 85% of churners" | Good retention strategy |
| Tenure = 40% importance | "Customers leave in first 3 months" | Improve onboarding |
| Monthly charges = 30% | "High bills cause churn" | Review pricing |
| F1 = 0.78 | "Model is solid, not perfect" | Use for top candidates, manual review for borderline |

### Writing Your Summary
Include:
1. **Best Model**: Which algorithm performed best?
2. **Metrics**: Accuracy, Precision, Recall, F1
3. **Top Drivers**: What causes churn?
4. **Recommendations**: What should company do?
5. **Impact**: How many customers identified per month?

---

## 🎯 Challenge Sequence

**Do them in order:**

1. **Setup & Load** → Prepare environment and data
2. **Explore** → Understand with Week 1 skills
3. **Visualize** → See relationships with churn
4. **Prepare** → Scale and split data
5. **Train** → Fit two models
6. **Predict** → Get predictions on test set
7. **Evaluate** → Calculate all metrics
8. **Confusion Matrix** → Interpret predictions
9. **Feature Importance** → Identify drivers
10. **Prediction Function** → Create reusable code
11. **ROC Curves** → Visualize quality
12. **Business Summary** → Communicate findings

---

## 💡 Common Mistakes to Avoid

❌ **Mistake 1**: Scaling before split
- You'll leak training info into test set!
- **Fix**: Fit scaler on training only, apply to test

❌ **Mistake 2**: Using class prediction for ROC curve
- ROC needs probabilities, not 0/1 predictions
- **Fix**: Use `.predict_proba()[:, 1]` instead

❌ **Mistake 3**: Ignoring class imbalance
- If 90% are no-churn, accuracy can be 90% while being useless
- **Fix**: Use Recall, F1, or ROC-AUC instead

❌ **Mistake 4**: Forgetting to document findings
- No one understands your beautiful model if not explained
- **Fix**: Write clear business recommendations

---

## 🚀 Ready?

You now understand every step. Time to build!

Open `STARTER_Weeks4-5_ChurnPredictor.ipynb` and tackle Challenge 1.

**Remember**: This is a real ML project. Companies do exactly this!

Good luck! 💪
