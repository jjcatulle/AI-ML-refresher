# Full ML Flow Checklist

Use this as your project ritual from start to finish.

Need a deeper, example-heavy version? Use `docs/ONE_PAGE_ML_FLOW_CARD.md` alongside this checklist.

## 0) Problem Definition
- Define the decision the model supports.
- State the target variable and prediction horizon.
- Write one baseline expectation.

Example:
```python
problem = "Predict customer churn in next billing cycle"
target_col = "Churn"
success_metric = "recall"  # choose based on business cost
baseline_rule = "predict majority class"
```

## 1) Data Understanding
- Confirm dataset source and scope.
- Inspect shape, schema, and key distributions.
- Identify risks: missingness, imbalance, drift-prone fields.

Example:
```python
print(df.shape)
print(df.dtypes)
print(df[target_col].value_counts(normalize=True))
print(df.isna().sum().sort_values(ascending=False).head(10))
```

## 2) Data Preparation
- Apply the Data Prep Cheatsheet.
- Lock preprocessing rules before model comparison.
- Document trade-offs and assumptions.

Example:
```python
X = df.drop(columns=[target_col])
y = (df[target_col].astype(str).str.lower() == "yes").astype(int)

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)
```

## 3) Exploratory Analysis and Visualization
- Plot distributions and target relationships.
- Check segment-level behavior (slices).
- Capture at least 3 observations that guide modeling choices.

Example:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=target_col, data=df)
plt.title("Target Distribution")
plt.show()
```

## 4) Baseline Model/System
- Train simplest viable baseline first.
- Record baseline metrics and limitations.
- Keep baseline as regression reference.

Example:
```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
print("Baseline F1:", f1_score(y_test, baseline_pred))
```

## 5) Model Training
- Train candidate models with controlled changes.
- Track hyperparameters and experiment notes.
- Avoid changing too many variables at once.

Example:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

## 6) Evaluation
- Use metrics aligned with business goal.
- Review confusion/error patterns, not only aggregate score.
- Compare against baseline, not against memory.

Example:
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
```

## 7) Improvement Loop
- Pick one improvement hypothesis.
- Re-run training and evaluate delta.
- Keep successful changes and revert regressions.

Example:
```python
hypothesis = "Lower threshold improves recall"
proba = model.predict_proba(X_test)[:, 1]
pred_t = (proba >= 0.40).astype(int)
# Compare metrics vs previous run and record delta
```

## 8) Validation and Reliability
- Test important slices and edge cases.
- Verify no obvious leakage or unstable behavior.
- Define known limitations explicitly.

Example:
```python
# Slice check example: tenure < 6 months
mask = X_test_raw["tenure"] < 6
print("Slice size:", mask.sum())
print("Slice target rate:", y_test[mask].mean())
```

## 9) Deployment or Handoff
- Prepare inference/preprocessing consistency notes.
- Package model, assumptions, and expected behavior.
- Add monitoring/eval plan for future updates.

Example:
```python
import joblib

joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model, "model.pkl")
# Save model card notes in markdown next to artifacts
```

## 10) Project Closeout
- Summarize what worked, what failed, what is next.
- Save final metrics, plots, and key decisions.
- Update tracker/checklist in docs.

Template:
```text
What improved:
What regressed:
Top risks remaining:
Next experiment:
```

## Weekly Habit Prompt
Before ending a session, write:
1. One thing improved today.
2. One risk still unresolved.
3. One next step for tomorrow.

## Companion Docs
- docs/DATA_PREP_CHEATSHEET.md
- docs/EVALUATION_FRAMEWORK.md
- guides/common/HOW_TO_SOLVE_CHALLENGES.md
- docs/AI-ML-LEARNING-GUIDE.md

Official references:
- https://scikit-learn.org/stable/user_guide.html
- https://scikit-learn.org/stable/common_pitfalls.html
- https://pandas.pydata.org/docs/
