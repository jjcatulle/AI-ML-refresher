# Guide: Fraud Detection Risk Scorer (Weeks 8-9)

## Beginner Start Here
This project teaches you how to build an ML model for an imbalanced, high-risk business problem.

### What this project does
Predict whether a transaction is fraudulent and produce a risk score you can threshold for operations.

### Terms you must know first
- `Class imbalance`: one class (fraud) is much rarer than the other.
- `Precision`: of flagged transactions, how many are truly fraud.
- `Recall`: of actual frauds, how many you caught.
- `PR-AUC`: area under precision-recall curve (best for imbalanced data).
- `Calibration`: whether predicted probabilities match true event frequencies.
- `Threshold`: cutoff used to convert probability to class label.
- `Cost matrix`: business penalty of false positives vs false negatives.

### Modules used
- `pandas`: load, clean, and transform transaction data.
- `numpy`: numeric transformations and vectorized metrics.
- `sklearn`: preprocessing, modeling, metrics, calibration, threshold sweeps.
- `matplotlib`/`seaborn`: distributions, confusion matrices, PR curves, reliability curves.

### How to study this guide
1. Build a simple baseline first.
2. Measure metrics that match fraud goals (precision/recall, PR-AUC).
3. Tune threshold based on business cost, not default 0.5.
4. Compare class-weighting vs resampling.
5. Add calibration and produce deployment-ready risk scoring logic.

---

## Big Picture
Fraud detection is usually a ranking + triage problem, not a pure yes/no prediction task.

You need:
- trustworthy probabilities,
- a clear threshold policy,
- repeatable evaluation,
- explanation of trade-offs.

**Why this matters:** In production, missing fraud can be more expensive than reviewing extra legitimate transactions. The model must align with business costs.

**Key Skills:**
- Handling severe class imbalance
- Choosing PR-focused metrics
- Threshold tuning with cost trade-offs
- Calibration and reliability checks
- Precision@k for analyst workflow

---

## Recommended Datasets

### Option 1: Kaggle Credit Card Fraud (Best for this project)
- Highly imbalanced binary classification dataset.
- Common benchmark for fraud workflows.
- Good for PR-AUC and threshold exercises.

### Option 2: IEEE-CIS Fraud Detection (Advanced)
- Larger and more realistic.
- Rich categorical and temporal features.
- Strong fit for feature engineering practice.

### Option 3: Synthetic Transactions (If you want full control)
- Generate known fraud patterns.
- Useful for debugging and explainability experiments.

---

## Concept 1: Why Accuracy Fails for Fraud

Suppose fraud rate is 0.5%.
A dumb model that predicts "not fraud" for every transaction gives 99.5% accuracy.
That is useless.

Use these instead:
- Precision
- Recall
- F1
- PR-AUC
- Precision@k

```python
from sklearn.metrics import precision_score, recall_score, average_precision_score
```

---

## Concept 2: Proper Train/Validation/Test Splits

For imbalanced data, use stratified splits so class ratios stay similar.

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
```

Rule:
- Train: fit model
- Validation: threshold and hyperparameter selection
- Test: final one-time report

---

## Concept 3: Baseline First

Always compare against a baseline before advanced methods.

Baseline options:
- Logistic Regression with `class_weight='balanced'`
- A simple rules baseline (e.g., high amount + unusual hour)

```python
from sklearn.linear_model import LogisticRegression

baseline = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
baseline.fit(X_train, y_train)
val_probs = baseline.predict_proba(X_val)[:, 1]
```

---

## Concept 4: Threshold Tuning (Core Skill)

Default threshold 0.5 is rarely optimal in fraud.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.arange(0.05, 0.96, 0.05)
rows = []
for t in thresholds:
    preds = (val_probs >= t).astype(int)
    rows.append({
        'threshold': t,
        'precision': precision_score(y_val, preds, zero_division=0),
        'recall': recall_score(y_val, preds, zero_division=0),
        'f1': f1_score(y_val, preds, zero_division=0)
    })
```

Choose threshold based on business constraint, for example:
- minimum recall >= 0.80
- maximize precision under that constraint

---

## Concept 5: PR Curve and PR-AUC

PR metrics are better than ROC when positives are rare.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

p, r, t = precision_recall_curve(y_val, val_probs)
pr_auc = average_precision_score(y_val, val_probs)
print('PR-AUC:', round(pr_auc, 4))
```

Interpretation:
- higher PR-AUC means better ability to find fraud without too many false alerts.

---

## Concept 6: Class Weighting vs Resampling

Two common imbalance strategies:
- `class_weight='balanced'`
- resampling (undersample majority or oversample minority)

Try both and compare with the same split.

Potential caveat:
- aggressive oversampling can overfit minority patterns.

---

## Concept 7: Confusion Matrix at Chosen Threshold

Confusion matrix is still useful once threshold is fixed.

```python
from sklearn.metrics import confusion_matrix

chosen_t = 0.22
preds = (val_probs >= chosen_t).astype(int)
cm = confusion_matrix(y_val, preds)
```

Read it with business impact:
- False Negatives: missed fraud (usually high cost)
- False Positives: manual review workload

---

## Concept 8: Calibration (Probability Quality)

A model can rank well but still output poorly calibrated probabilities.

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(baseline, method='isotonic', cv=3)
calibrated.fit(X_train, y_train)
cal_probs = calibrated.predict_proba(X_val)[:, 1]
```

Reliability curves help verify if predicted 0.2 really means ~20% fraud rate.

---

## Concept 9: Precision@k for Operations

Fraud analysts often review top-k risky transactions, not all predictions.

```python
import numpy as np

k = 200
idx = np.argsort(-val_probs)[:k]
precision_at_k = y_val.iloc[idx].mean()
print('Precision@k:', round(float(precision_at_k), 4))
```

This metric directly maps to review queue quality.

---

## Concept 10: Cost-Aware Decisioning

Define a simple cost function:
- FN cost = 25 units
- FP cost = 1 unit

Select threshold minimizing expected cost on validation.

```python
FN_COST = 25
FP_COST = 1

def total_cost(y_true, y_pred):
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return FP_COST * fp + FN_COST * fn
```

This is often better than optimizing F1 blindly.

---

## Suggested Project Workflow

1. Load and audit data quality.
2. Build baseline model.
3. Evaluate PR-AUC and recall/precision.
4. Tune threshold with business constraints.
5. Compare imbalance strategies.
6. Calibrate best model.
7. Build risk scoring function.
8. Write fraud ops summary.

---

## Reflection Questions

1. Why is PR-AUC preferred over accuracy for fraud?
2. What threshold did you choose and why?
3. Which is more expensive in your scenario: false positives or false negatives?
4. Did class weighting or resampling perform better on validation?
5. How did calibration change your probability interpretation?
6. If you can review only top 200 transactions daily, what metric matters most?

---

## Checklist

- [ ] I used stratified splits for train/val/test.
- [ ] I built at least one baseline model.
- [ ] I evaluated precision, recall, F1, PR-AUC.
- [ ] I swept thresholds and chose one based on business logic.
- [ ] I compared at least two imbalance strategies.
- [ ] I plotted/checked calibration.
- [ ] I computed precision@k.
- [ ] I wrote a risk scoring function and plain-language summary.

---

*Guide for `phases/phase1/starters/STARTER_Weeks8-9_FraudDetection.ipynb` | Phase 1 | ML-AI-learning roadmap*
