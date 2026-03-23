# Week 1: Data Exploration (Deep Dive)

## 1. Core Concepts
- EDA pipeline:
  1. Understand data semantics (columns, units, business definitions).
  2. Validate integrity (type checks, duplicates, missingness, distribution).
  3. Log transformations and outliers for robust analysis.
  4. Communicate insights with summaries and plots.
- Data typing:
  - Quantitative: continuous vs discrete.
  - Qualitative: nominal vs ordinal.
  - Special: timestamps, text, geospatial.
- Statistical measures:
  - Central tendency: mean, median, mode.
  - Dispersion: variance `σ²`, standard deviation `σ`, IQR.
  - Shape: skewness and kurtosis, using `scipy.stats`.

## 2. Suggested EDA script structure
1. `df = pd.read_csv('filename.csv', parse_dates=[...], low_memory=False)`.
2. `df.info(); df.describe(include='all'); df.nunique()`.
3. Null patterns: `msno.matrix(df)` and `missing = df.isna().mean()`.
4. Duplicate detection: `df.duplicated().sum()`.
5. Outlier detection:
   - Univariate via IQR and `|Z| > 3`.
   - Multivariate using Mahalanobis distance.
6. Correlations:
   - Numeric: Pearson, Spearman, Kendall.
   - Categorical: Cramér’s V.
   - Mixed: point-biserial, numeric encoding.

## 3. Visual exploration (with code)
- Distribution plots:
  - `sns.histplot(data=df, x='feature', kde=True)`
  - `sns.boxplot(y='feature', data=df)`
- Relationships:
  - Scatter with regression: `sns.lmplot(x='x', y='y', data=df)`.
  - Pairwise: `sns.pairplot(df[numerics])`.
  - Heatmap: `sns.heatmap(df.corr(), annot=True)`.
- Categorical:
  - `pd.crosstab(df['cat1'], df['cat2'], normalize='index')`.
  - Bar chart for frequency and class balance.

## 4. Advanced topics and decision logic
### 4.1 Imputation strategy decision
- Numeric:
  - low missing <5%: mean/median.
  - moderate 5-20%: KNN, IterativeImputer.
  - high >20%: domain analysis, consider removal.
- Categorical:
  - mode, frequent class, new category (`'missing'`), or conditional impute.

### 4.2 Feature engineering and transformation
- Date features: age, tenure, weekday/weekend, seasonal buckets.
- Binning: `pd.qcut` for quantiles, `pd.cut` with custom bins.
- Interaction: `feat_1 * feat_2`, polynomial terms with `PolynomialFeatures`.
- Target-guided: mean target encoding with smoothing.

### 4.3 Imbalance strategy
- Class weights in models: `class_weight='balanced'` (sklearn)
- Resampling techniques: SMOTE / BorderlineSMOTE / ADASYN.
- Evaluate using stratified k-fold.

## 5. Tools and libraries (deep)
- Visual EDA: `pandas-profiling` / `ydata-profiling`, `sweetviz`, `dtale`.
- Statistical tests: `scipy.stats` (KS-test, chi2, ANOVA), `statsmodels` OLS.
- Auto feature analysis: `featuretools`, `sklearn.feature_selection`.

## 6. References
- Wes McKinney, "Python for Data Analysis".
- Joel Grus, "Data Science from Scratch" (EDA chapter).
- Practical case study: "The Data Science Process" on Kaggle.
- Article: "EDA done right" (medium.com).

## 7. Math-to-Code Bridge — Manual Implementation Day ⭐
Before trusting libraries, implement the core algorithms yourself with raw NumPy. This teaches you how the dimensions align and what the library is actually doing.

### Why this matters
`LinearRegression().fit(X, y)` is one line of code — but if you don't know why `X` must be shape `[n_samples, n_features]`, you will get mysterious errors and have no idea how to debug them.

### The Normal Equation (Closed-form solution for Linear Regression)
Linear regression finds weights θ that minimize sum of squared errors. The exact answer (no iterations needed) is:

`θ = (X^T X)^{-1} X^T y`

Dimension walkthrough:
- `X`: shape `[n, p]` — n samples, p features
- `X^T`: shape `[p, n]`
- `X^T X`: shape `[p, p]` — square, invertible (if features aren't duplicated)
- `(X^T X)^{-1}`: shape `[p, p]`
- `X^T y`: shape `[p, 1]`
- `θ`: shape `[p, 1]` — one weight per feature ✓

```python
import numpy as np

# Manual Linear Regression via Normal Equation
np.random.seed(42)
n, p = 100, 3
X = np.random.randn(n, p)
true_theta = np.array([2.0, -1.0, 0.5])
y = X @ true_theta + np.random.randn(n) * 0.1  # add small noise

# Add bias column (column of 1s) so model learns intercept
X_b = np.c_[np.ones(n), X]   # shape: [100, 4]

# Normal equation
theta_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print('Learned:', theta_hat[1:])  # should be close to [2, -1, 0.5]
print('Bias:',    theta_hat[0])   # intercept

# Compare against sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
print('sklearn coef:', model.coef_)  # should match
```

### Gradient Descent (Iterative solution)
Use when data is too large to invert `X^T X` (inversion is O(p³)). Instead, take small steps in the direction that reduces error.

Update rule: `θ = θ - α * (1/n) * X^T (Xθ - y)`

- `α` (alpha) = learning rate: how big each step is
- `X^T (Xθ - y)` = gradient (direction of steepest increase in error — we subtract to go down)

```python
# Manual Gradient Descent for Linear Regression
alpha = 0.01   # learning rate
n_iters = 1000
theta = np.zeros(X_b.shape[1])  # start at zeros

loss_history = []
for i in range(n_iters):
    y_pred = X_b @ theta               # predictions: [n]
    error  = y_pred - y                # residuals: [n]
    grad   = (1/n) * X_b.T @ error    # gradient: [p]
    theta  = theta - alpha * grad      # update step

    if i % 100 == 0:
        loss = np.mean(error**2)       # MSE
        loss_history.append(loss)
        print(f'Step {i}: MSE = {loss:.4f}')

print('GD result:', theta[1:])    # should match normal equation
```

Key insight: Watch the loss decrease and plateau. If learning rate is too high, loss explodes. Too low, convergence is slow. This is why tuning `lr` matters.

## 8. Challenge
- Build an interactive EDA notebook that:
  1. Loads dataset and creates data quality report.
  2. Performs 2 types of decomposition (PCA, t-SNE) and compares cluster goodness.
  3. Benchmarks 4 imputation strategies with cross-validation impact on a baseline model.
  4. Writes a summary with next-action recommendations for feature engineering.
- **Math-to-Code:** Implement both the Normal Equation and Gradient Descent from scratch. Verify they converge to the same θ as `sklearn.LinearRegression`. Plot MSE vs iteration for Gradient Descent.

