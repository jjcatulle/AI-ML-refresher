# One-Page ML Flow Card (Expanded Practical Edition)

Use this as your default workflow for every tabular ML project.

## How to use this card
- Spend 2 to 5 minutes on each section before moving on.
- Do not skip sections because your model "already works".
- Keep a short experiment log while following the flow.

## 0) Problem framing
### What to write
- Decision to support: what action this model changes.
- Target: exact column and label definition.
- Prediction window: when the prediction is made vs when outcome is known.
- Cost of errors: false positive cost vs false negative cost.

### Example template
```text
Decision: flag customers for retention campaign.
Target: churn (1 if canceled in next billing cycle, else 0).
Prediction time: day 25 of cycle.
Outcome known: after cycle closes.
Error costs: FN > FP because missed churn is expensive.
Primary metric: recall with minimum precision floor.
```

## 1) Data contract and scope
### Checklist
- Confirm dataset owner, extraction time, and source system.
- Freeze scope columns for this iteration.
- Define exclusion rules (test accounts, internal users, invalid IDs).

### Example code
```python
required_cols = {
    "customer_id", "tenure", "monthly_charges", "contract_type", "churn"
}
missing_required = required_cols - set(df.columns)
assert not missing_required, f"Missing required columns: {missing_required}"

# Remove obvious out-of-scope rows
if "is_test_account" in df.columns:
    df = df[df["is_test_account"] == 0].copy()
```

## 2) Data quality pass
### Checks
- Null profile by column and by important segment.
- Duplicate profile (full row and key-based).
- Type sanity (numeric in object columns, parseable dates).
- Range/business rules (negative age, impossible spend, future timestamps).

### Example code
```python
print(df.isna().mean().sort_values(ascending=False).head(10))
print("Full-row duplicates:", df.duplicated().sum())
print("ID duplicates:", df.duplicated(subset=["customer_id"]).sum())

# Range checks
if "tenure" in df.columns:
    print("Invalid tenure rows:", (df["tenure"] < 0).sum())
```

## 3) Target and leakage review
### Checklist
- Verify target is not directly encoded in features.
- Remove post-outcome columns (anything only known after event).
- Confirm time consistency for time-aware datasets.

### Example leakage checks
```python
leak_risk_cols = [
    c for c in df.columns
    if any(k in c.lower() for k in ["cancel", "closed", "final", "outcome"])
]
print("Potential leakage columns:", leak_risk_cols)
```

## 4) Split strategy
### Rules
- Split before fitting transformations.
- Use stratification for imbalanced classification.
- Use time-based split for temporal prediction problems.
- Keep random seed fixed for reproducibility.

### Example code
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["churn"])
y = df["churn"].astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## 5) Preprocessing pipeline
### Goal
Make transformations reproducible and identical across train, test, and inference.

### Example code
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = X_train_raw.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train_raw.select_dtypes(exclude=["number"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])
```

## 6) Baseline first
### Checklist
- Always benchmark with a simple baseline.
- Record baseline metrics in your experiment notes.
- Model improvement is valid only if it beats baseline on selected metric.

### Example code
```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score, precision_score

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_raw, y_train)
b_pred = baseline.predict(X_test_raw)

print("Baseline recall:", recall_score(y_test, b_pred))
print("Baseline precision:", precision_score(y_test, b_pred, zero_division=0))
```

## 7) Train candidate model
### Example code
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])

model.fit(X_train_raw, y_train)
pred = model.predict(X_test_raw)
```

## 8) Evaluate deeply
### Checklist
- Report primary metric and supporting metrics.
- Inspect confusion matrix.
- Review calibration/threshold behavior when relevant.
- Compare directly against baseline.

### Example code
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(classification_report(y_test, pred, digits=3))
print(confusion_matrix(y_test, pred))

proba = model.predict_proba(X_test_raw)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
```

## 9) Improve with one hypothesis at a time
### Good hypotheses
- Threshold tuning for recall/precision tradeoff.
- Feature set cleanup (remove noisy columns).
- Class-weight adjustments for imbalance.
- Simpler model for stability and interpretability.

### Example threshold scan
```python
import numpy as np
from sklearn.metrics import f1_score

for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (proba >= t).astype(int)
    print(t, "recall", recall_score(y_test, p), "f1", f1_score(y_test, p))
```

## 10) Slice and robustness checks
### Checklist
- Validate performance across important slices.
- Check for unstable behavior on small groups.
- Document known weak slices.

### Example code
```python
slice_masks = {
    "tenure_lt_6": X_test_raw["tenure"] < 6,
    "high_monthly": X_test_raw["monthly_charges"] >= 80,
}

for name, mask in slice_masks.items():
    if mask.sum() == 0:
        continue
    print(name, "size", int(mask.sum()), "recall", recall_score(y_test[mask], pred[mask]))
```

## 11) Package and handoff
### Checklist
- Save full pipeline, not only the model.
- Record schema expectations and feature assumptions.
- Add a minimal model card: data range, metrics, limitations, owner.

### Example code
```python
import joblib

joblib.dump(model, "models/churn_pipeline.pkl")

# Optional: simple metadata file
metadata = {
    "target": "churn",
    "test_recall": float(recall_score(y_test, pred)),
    "random_state": 42,
}
print(metadata)
```

## 12) Closeout reflection (mandatory)
### Write this after each project
```text
What improved most vs baseline:
Largest unresolved risk:
Most likely source of bias:
One change to test next:
```

## Quick metric guide
- Use recall-first when missing positives is expensive.
- Use precision-first when acting on false positives is expensive.
- Use ROC-AUC/PR-AUC for ranking quality.
- Keep at least one thresholded metric tied to business action.

## Common failure patterns
- Fitting preprocessing on full data before split.
- Comparing models with different data subsets.
- Tracking only one metric in an imbalanced problem.
- Ignoring feature leakage from post-outcome signals.
- Skipping slice checks and shipping global metrics only.

## Internal references
- docs/DATA_PREP_CHEATSHEET.md
- docs/FULL_ML_FLOW_CHECKLIST.md
- docs/EVALUATION_FRAMEWORK.md
- docs/AI-ML-LEARNING-GUIDE.md
- guides/common/HOW_TO_SOLVE_CHALLENGES.md

## Official references
- Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Scikit-learn Common Pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
- Scikit-learn Model Selection: https://scikit-learn.org/stable/model_selection.html
- Scikit-learn Pipeline and Composite Estimators: https://scikit-learn.org/stable/modules/compose.html
- Scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Pandas User Guide: https://pandas.pydata.org/docs/user_guide/index.html
- Feature engineering best practices (Google): https://developers.google.com/machine-learning/guides/rules-of-ml
