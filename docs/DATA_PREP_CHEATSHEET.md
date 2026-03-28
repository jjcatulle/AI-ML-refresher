# Data Prep Cheatsheet

Use this before visualization or training in every project.

For a full end-to-end execution guide with richer examples and references, see `docs/ONE_PAGE_ML_FLOW_CARD.md`.

## Goal
Turn raw data into reliable, model-ready input with repeatable steps.

## 1) Define Scope
- What is the target variable?
- What prediction/decision will this model support?
- Which rows/columns are in scope and out of scope?

Example:
```python
target_col = "Churn"
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col].copy()
print("X shape:", X.shape, "| y shape:", y.shape)
```

## 2) Run Quality Checks
- Missing values: count per column and pattern by segment.
- Duplicates: exact and key-based duplicates.
- Data types: numeric fields stored as strings, date parsing issues.
- Range checks: impossible values, outliers, unit mismatches.

Example:
```python
print("Missing values:\n", df.isna().sum().sort_values(ascending=False).head(10))
print("Duplicate rows:", df.duplicated().sum())
print("Dtypes:\n", df.dtypes)

# Quick numeric range sanity check
num_cols = df.select_dtypes(include=["number"]).columns
print(df[num_cols].describe().T[["min", "max"]].head(10))
```

## 3) Fix Core Issues
- Standardize column names and categories.
- Convert data types intentionally.
- Handle missing values with explicit strategy:
  - Drop only with reason.
  - Impute with method appropriate to column type.
- Keep a short note of every fix and why it was chosen.

Example:
```python
# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Convert known numeric-like strings
for col in ["monthlycharges", "totalcharges"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Standardize target if text labels
if "churn" in df.columns and df["churn"].dtype == "object":
    df["churn"] = (df["churn"].str.strip().str.lower() == "yes").astype(int)
```

## 4) Prepare Features
- Split features into numeric and categorical groups.
- Encode categorical values consistently.
- Scale numeric features when model sensitivity requires it.
- Prevent leakage: never use future information in feature creation.

Example:
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = df.drop(columns=["churn"])
y = df["churn"]

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
  ("imputer", SimpleImputer(strategy="median")),
  ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
  ("num", numeric_pipe, numeric_cols),
  ("cat", categorical_pipe, categorical_cols),
])
```

## 5) Split Before Fitting Prep
- Create train/test split first.
- Fit preprocessing only on train data.
- Apply the same fitted preprocessing to test/inference data.

Example:
```python
from sklearn.model_selection import train_test_split

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

print("Train matrix:", X_train.shape, "| Test matrix:", X_test.shape)
```

## 6) Readiness Checks
- Target is clean and consistently encoded.
- Train and test pass through identical preprocessing logic.
- No unresolved blockers in critical fields.
- Feature matrix shape and class balance are understood.

Example:
```python
print("Target dtype:", y_train.dtype)
print("Target distribution:\n", y_train.value_counts(normalize=True).round(3))
print("Nulls in train raw (top 10):\n", X_train_raw.isna().sum().sort_values(ascending=False).head(10))

assert set(y_train.unique()).issubset({0, 1}), "Target should be binary 0/1"
assert X_train.shape[0] == y_train.shape[0], "Row mismatch between X_train and y_train"
```

## 7) Common Mistakes to Avoid
- Cleaning the full dataset before splitting.
- Using different transformations in train vs inference.
- Dropping many rows without impact review.
- Relying on accuracy only for imbalanced tasks.

## 8) Quick Reflection
Write 3 short answers before training:
1. What did I fix and why?
2. What assumptions did I make?
3. What prep decision is most likely to affect model behavior?

Template:
```text
Fixes made:
Assumptions:
Highest-risk prep decision:
```

## References
- guides/phase0/GUIDE_Week2_PandasDeep.md
- guides/phase0/GUIDE_Week3_NumPyDeep.md
- guides/phase0/GUIDE_Week4_Visualization.md
- docs/AI-ML-LEARNING-GUIDE.md

Official references:
- https://pandas.pydata.org/docs/
- https://scikit-learn.org/stable/modules/preprocessing.html
- https://scikit-learn.org/stable/modules/compose.html#column-transformer
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://scikit-learn.org/stable/common_pitfalls.html
