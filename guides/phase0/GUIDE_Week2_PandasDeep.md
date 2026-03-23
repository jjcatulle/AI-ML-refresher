# Guide: Week 2 — pandas Deep Dive
> **Phase 0 | Foundation** — Master the tool you'll use in every project.

---

## Beginner Start Here

**pandas** is the workhorse of data science. In a 2026 ML engineer role, you'll use pandas (or its faster cousin polars) every single day. This guide explains not just what the pandas functions do, but **why** you use them and **when** to reach for each one.

### What This Guide Covers
- The `Series` and `DataFrame` data model
- Loading, inspecting, and understanding a new dataset
- Selecting, filtering, and slicing data
- Handling missing values (the right way)
- Transforming and creating new columns
- Aggregating with `groupby`
- The full EDA workflow

### Key Terms
| Term | Plain English |
|------|---------------|
| **DataFrame** | A 2D table: rows × columns. Like a spreadsheet that lives in memory |
| **Series** | A 1D labeled array — one column of a DataFrame |
| **Index** | Row identifiers (default: 0, 1, 2...). Can be set to any unique column |
| **NaN** | Not a Number — how pandas represents missing data |
| **dtype** | Column data type: int64, float64, object (string), bool, category, datetime |
| **Vectorized operation** | Applying math/logic to an entire column simultaneously — no loop needed |
| **Mask** | A boolean Series used to filter rows |
| **GroupBy** | Split the DataFrame into groups, apply a function to each group |
| **Aggregation** | Reducing many values to one: sum, mean, count, min, max |

---

## How to Study This Guide

1. Read Section 1 (Series/DataFrame model). Draw a diagram on paper.
2. Open `STARTER_Week2_PandasDeep.ipynb`. Run the first "Create a DataFrame" cell.
3. For each section in this guide, read it, then run the matching notebook section.
4. Do **not** copy code from this guide into the notebook — type it yourself.
5. After finishing, do the full EDA workflow exercise at the bottom of the notebook.

---

## Section 1: The pandas Data Model

### Series — One Column

```python
import pandas as pd

s = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])
```

```
a    10.0
b    20.0
c    30.0
dtype: float64
```

A Series has:
- **Values** — the actual data (`[10.0, 20.0, 30.0]`)
- **Index** — the labels for each value (`["a", "b", "c"]`)
- **dtype** — the type of all values (`float64`)
- **name** — optional name (same as the column name when part of a DataFrame)

### DataFrame — A Table

```
   customer_id  monthly_charges  contract          churned
0  C001         65.00            Month-to-month    0
1  C002         110.00           Month-to-month    1
2  C003         45.00            Two year          0
```

A DataFrame has:
- **Columns** — each is a Series
- **Index** — row labels (integers by default)
- **Shape** — `(n_rows, n_cols)` e.g., `(1000, 12)`

**Key insight:** Every column operation in pandas (e.g., `df['charges'] * 0.9`) applies the math to **all rows at once** — this is what "vectorized" means. Never loop over rows when you can use column operations.

---

## Section 2: Inspecting a New Dataset

You should run these **every single time** you open a new CSV or database table. Make this a habit.

```python
# Step 1: Load
df = pd.read_csv("data.csv")

# Step 2: Shape
print(df.shape)          # (rows, columns)

# Step 3: Preview
df.head(5)               # first 5 rows
df.tail(5)               # last 5 rows
df.sample(5)             # 5 random rows

# Step 4: Structure
df.info()                # column names, non-null counts, dtypes

# Step 5: Stats
df.describe()            # mean, std, min, quartiles, max for numeric cols
df.describe(include='all')  # includes string columns too

# Step 6: Missing
df.isnull().sum()        # missing count per column
df.isnull().mean()       # missing proportion per column (0.0 to 1.0)

# Step 7: Target balance (for classification)
df['target'].value_counts(normalize=True)  # class proportions
```

### What to Look For

| Issue | How to Spot It | Why It Matters |
|-------|----------------|----------------|
| Missing values | `df.isnull().sum()` | Most ML algorithms can't handle NaN |
| Wrong dtype | `df.info()` shows `object` for a numeric col | Must convert before modeling |
| Outliers | `df.describe()` — max much larger than 75th pct | Can distort linear models |
| Class imbalance | `df['target'].value_counts(normalize=True)` | Accuracy will be misleading |
| Duplicate rows | `df.duplicated().sum()` | Inflates model confidence |
| Constant column | `df['col'].nunique() == 1` | Useless feature, drop it |

---

## Section 3: Selecting Data

### Column Selection

```python
# One column → Series
df['column_name']

# Multiple columns → DataFrame
df[['col1', 'col2', 'col3']]

# All column names
df.columns.tolist()

# Columns by dtype
df.select_dtypes(include='number')   # numeric cols only
df.select_dtypes(include='object')   # string cols only
```

### Row Selection: `.loc` vs `.iloc`

| | `.loc` | `.iloc` |
|--|--------|---------|
| Selects by | Label (index value) | Position (integer 0-based) |
| Row syntax | `df.loc[row_label]` | `df.iloc[row_position]` |
| Range | `df.loc[0:5]` **(inclusive)** | `df.iloc[0:5]` **(exclusive)** |
| Slice with cols | `df.loc[0:5, 'col':'col2']` | `df.iloc[0:5, 0:3]` |

**Rule of thumb:** Use `.loc` when you know column names. Use `.iloc` when you know position numbers.

---

## Section 4: Boolean Filtering (Boolean Masking)

This is the most powerful row selection technique. It creates a boolean Series and uses it to filter.

```python
# Step 1: Create a mask (boolean Series)
mask = df['monthly_charges'] > 75

# Step 2: Apply the mask
df[mask]                         # rows where mask is True
df[df['monthly_charges'] > 75]  # same thing, one line
```

### Combining Conditions

```python
# AND — both conditions must be True
df[(df['charges'] > 75) & (df['churned'] == 1)]

# OR — at least one condition must be True
df[(df['charges'] > 100) | (df['tenure'] < 3)]

# NOT — negate a condition
df[~(df['contract'] == 'Month-to-month')]  # all contracts except M-to-M

# Multiple values — .isin()
df[df['contract'].isin(['Month-to-month', 'One year'])]

# String contains — .str.contains()
df[df['email'].str.contains('@gmail.com', na=False)]
```

### Common Mistake: Using `and`/`or` Instead of `&`/`|`

```python
# WRONG — Python's 'and' doesn't work on Series:
df[df['a'] > 3 and df['b'] < 5]  # ValueError!

# CORRECT — use bitwise operators:
df[(df['a'] > 3) & (df['b'] < 5)]  # ✅
```

---

## Section 5: Handling Missing Values

Missing values (`NaN`) are the most common data quality issue. You must deal with them before training any model.

### Decision Tree: What Strategy to Use?

```
Is the column very important for the model?
  → Yes → Fill with a statistic (don't drop rows)
  → No  → Could drop the column

Is the column numeric?
  → Yes → Fill with median (robust to outliers) or mean
  → No  → Fill with mode (most common value) or 'unknown'

How much data is missing?
  → < 5%  → Safe to fill or drop rows
  → 5-30% → Fill carefully, add a binary 'was_missing' flag column
  → > 30% → Consider dropping the column entirely
```

### pandas Methods

```python
# Check
df.isnull().sum()
df['col'].isna()      # alias for isnull()

# Fill
df['col'].fillna(df['col'].median())   # fill with median
df['col'].fillna(df['col'].mode()[0])  # fill with mode
df['col'].fillna(0)                    # fill with constant
df['col'].fillna(method='ffill')       # forward fill (time series)

# Drop
df.dropna()                      # drop any row with a NaN
df.dropna(subset=['col'])        # drop rows where 'col' is NaN
df.dropna(thresh=5)              # drop rows with fewer than 5 non-null values

# Add "was_missing" flag (preserves information)
df['col_was_missing'] = df['col'].isnull().astype(int)

# ALWAYS work on a copy!
df_clean = df.copy()
df_clean['col'] = df_clean['col'].fillna(median)
```

---

## Section 6: Transforming Columns

### Creating New Columns

```python
# Math operations
df['total_charges'] = df['monthly_charges'] * df['tenure_months']

# Conditional: np.where
import numpy as np
df['high_risk'] = np.where(df['monthly_charges'] > 90, 1, 0)

# Binning: pd.cut
df['tenure_band'] = pd.cut(df['tenure_months'],
                            bins=[0, 6, 12, 24, 100],
                            labels=['0-6m', '7-12m', '1-2yr', '2yr+'])

# Map (encode categories)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df['contract_enc'] = df['contract'].map(contract_map)

# Apply (custom function to each value)
df['name_upper'] = df['name'].apply(str.upper)
df['score_label'] = df['score'].apply(lambda x: 'pass' if x >= 0.5 else 'fail')
```

### String Column Operations with `.str`

```python
df['email'].str.lower()             # lowercase all
df['name'].str.split(' ').str[0]    # first word
df['contract'].str.replace('-', '_')
df['desc'].str.contains('premium')  # boolean Series
df['text'].str.len()                # string length
```

### Renaming and Dropping

```python
# Rename
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# Drop columns
df.drop(columns=['col1', 'col2'], inplace=True)

# Drop rows by index
df.drop(index=[0, 5, 10], inplace=True)
```

---

## Section 7: GroupBy — Your Most Powerful Analysis Tool

GroupBy is the pandas equivalent of SQL's `GROUP BY`. It lets you answer questions like:
- "What is the average charges for each contract type?"
- "What is the churn rate by tenure band?"

### The Three Steps

```python
# 1. Split: group the DataFrame
groups = df.groupby('contract')

# 2. Apply: compute something per group
result = groups['monthly_charges'].mean()

# 3. Combine: pandas returns a Series or DataFrame
```

### Common Aggregations

```python
df.groupby('contract')['charges'].mean()       # one function, one column
df.groupby('contract')['charges'].agg(['mean', 'median', 'std', 'count'])

# Multiple columns
df.groupby('contract').agg(
    n_customers=('customer_id', 'count'),
    avg_charges=('monthly_charges', 'mean'),
    churn_rate=('churned', 'mean'),
    median_tenure=('tenure_months', 'median')
)
```

### GroupBy for Churn Analysis (ML Use Case)

```python
# Which features have different means between churned and active customers?
df.groupby('churned')[['monthly_charges', 'tenure_months', 'support_tickets']].mean()
```

This tells you which features are most predictive of churn. Features with the largest difference in group means tend to have the most predictive power.

---

## Section 8: Common Patterns in ML Data Prep

### The Standard Preprocessing Pipeline

```python
import pandas as pd
import numpy as np

def prepare_churn_data(filepath):
    """Full data prep pipeline — returns X, y ready for sklearn."""
    df = pd.read_csv(filepath)
    
    # 1. Drop useless columns
    df = df.drop(columns=['customer_id'])
    
    # 2. Handle missing values
    df['tenure_months'] = df['tenure_months'].fillna(df['tenure_months'].median())
    
    # 3. Feature engineering
    df['total_charges'] = df['monthly_charges'] * df['tenure_months']
    
    # 4. Encode categoricals
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df['contract_enc'] = df['contract'].map(contract_map)
    df = df.drop(columns=['contract'])
    
    # 5. Split features and target
    X = df.drop(columns=['churned'])
    y = df['churned']
    
    return X, y

X, y = prepare_churn_data('data/churn.csv')
print(f"X shape: {X.shape}, y shape: {y.shape}")
```

### Reading and Writing

```python
# Read
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', index_col=0)   # use first col as index
df = pd.read_json('file.json')
df = pd.read_excel('file.xlsx')

# Write
df.to_csv('output.csv', index=False)        # don't write row numbers
df.to_parquet('output.parquet')             # faster, smaller file (use in production)
```

---

## pandas vs SQL Reference

If you know SQL, this maps directly:

| SQL | pandas |
|-----|--------|
| `SELECT col1, col2 FROM df` | `df[['col1', 'col2']]` |
| `WHERE col > 5` | `df[df['col'] > 5]` |
| `ORDER BY col DESC` | `df.sort_values('col', ascending=False)` |
| `GROUP BY col` | `df.groupby('col')` |
| `COUNT(*)` | `.count()` or `len(df)` |
| `AVG(col)` | `.mean()` |
| `JOIN` | `pd.merge(df1, df2, on='col')` |
| `LIMIT 10` | `df.head(10)` |

---

## Reflection Questions

1. What does `df.info()` show that `df.describe()` doesn't?
2. Why do we use `df.copy()` before modifying a DataFrame?
3. What is the difference between `.loc[0:5]` and `.iloc[0:5]`?
4. Why fill missing numeric values with the median instead of the mean?
5. Write from memory: the code to find the average churn rate per contract type.
6. What does `value_counts(normalize=True)` return?

---

## Checklist for This Week

- [ ] I can load a CSV and run the full inspection sequence
- [ ] I can select single columns, multiple columns, and filter rows
- [ ] I understand the difference between `.loc` and `.iloc`
- [ ] I can find, fill, and drop missing values
- [ ] I can create new columns from existing ones with math and `.map()`
- [ ] I can group data and compute aggregations with `.groupby().agg()`
- [ ] I completed the full EDA workflow notebook exercise
- [ ] I saved a cleaned DataFrame to CSV

---

*Guide for `STARTER_Week2_PandasDeep.ipynb` | Phase 0 | ML-AI-learning roadmap*
