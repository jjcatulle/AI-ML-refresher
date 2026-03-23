# Weeks 6-7: Risk Scoring with Linear Models (Deep Dive)

## 1. What is Risk Scoring?
Risk scoring is a way to predict how likely something bad will happen, like a loan default, equipment failure, or health issue. We use a number (score) between 0 and 1, where 0 means "very low risk" and 1 means "very high risk". This score helps businesses make decisions, like approving loans or scheduling maintenance.

- **Why linear models?** They are simple, fast to train, and easy to understand. They work well when relationships between features (inputs) and risk are mostly straight-line (linear).
- **Key terms:**
  - **Regression:** Predicting a continuous number (like risk score) instead of categories (like yes/no).
  - **Calibration:** Making sure the score matches real-world probabilities (e.g., if score is 0.7, it should happen 70% of the time).
  - **Threshold:** A cutoff point, like "if score > 0.5, flag as high risk".

## 2. Problem Setup and Data Prep
- **Define the target:** What are you predicting? For example, "probability of customer defaulting on a loan in the next 6 months".
- **Collect features:** Things that might affect risk, like income, credit history, age. Clean data by removing duplicates and filling missing values.
- **Feature engineering:** Create new features from existing ones.
  - **Volatility:** How much something changes over time (e.g., income fluctuation).
  - **Streaks:** How many times in a row something happened (e.g., late payments).
  - **Ratio metrics:** Comparisons like "debt-to-income ratio".
  - **Seasonal lags:** Values from previous periods (e.g., last month's sales).
- **Handle multicollinearity:** When features are too similar, it confuses the model. Use Variance Inflation Factor (VIF) to detect and remove them.

## 3. Building Linear Models
- **Basic linear regression:** Assumes a straight line: `risk_score = w1*feature1 + w2*feature2 + ... + b`.
  - Train with `sklearn.linear_model.LinearRegression()`.
- **Ridge regression:** Adds penalty to prevent overfitting (when model memorizes training data too well). Use `Ridge(alpha=0.1)`.
- **Lasso regression:** Like Ridge but can set some weights to zero, doing automatic feature selection. Use `Lasso(alpha=0.01)`.
- **ElasticNet:** Mix of Ridge and Lasso. Use `ElasticNet(alpha=0.01, l1_ratio=0.5)`.
- **Scaling:** Features need to be on similar scales. Use `StandardScaler()` (subtract mean, divide by std) or `RobustScaler()` (ignores outliers).
- **Adding non-linearities:** If relationships aren't straight, transform features: `np.log(feature)` for exponential data, `np.sqrt(feature)` for quadratic.

## 4. Training and Validation
- **Split data:** Use 70% for training, 20% for validation, 10% for testing. Never peek at test data until end.
- **Cross-validation:** Train on subsets to check stability. Use `cross_val_score(model, X, y, cv=5)`.
- **Diagnostics:**
  - **Residuals:** Differences between predicted and actual values. Plot them to check for patterns (should be random).
  - **Homogeneity:** Variance of residuals should be constant (no funnel shape).
  - **Normality:** Residuals should follow a bell curve.
  - **Independence:** No correlation between residuals.
- **Metrics:**
  - **RSS (Residual Sum of Squares):** Sum of squared errors.
  - **RMSE (Root Mean Squared Error):** Average error size.
  - **MAE (Mean Absolute Error):** Average absolute error.
  - **R²:** How much variance the model explains (0 to 1, higher is better).
  - **Adjusted R²:** Penalizes adding useless features.

## 5. Advanced Topics
- **GLM (Generalized Linear Models):** For different data types.
  - **Poisson:** For counts (e.g., number of accidents).
  - **Binomial:** For yes/no outcomes.
  - Use `statsmodels` library: `sm.GLM(y, X, family=sm.families.Binomial()).fit()`.
- **Quantile regression:** Predicts percentiles, not just averages. Useful for risk ranges.
- **Fairness:** Check if model treats groups equally. Measure disparate impact: does it unfairly flag certain groups?

## 6. Code Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('risk_data.csv')
X = df.drop('risk_score', axis=1)
y = df['risk_score']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Train model
model = Ridge(alpha=0.1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')
```

## 7. References
- Book: "Introduction to Statistical Learning" by Hastie et al. (free online).
- Tutorial: Scikit-learn docs on linear models.
- Video: StatQuest YouTube series on regression.

## 8. Challenge
- Build a risk model for a dataset (e.g., credit risk).
- Compare Ridge, Lasso, and ElasticNet on RMSE.
- Calibrate scores to probabilities using `CalibratedClassifierCV`.
- Measure Brier score (how well probabilities match reality).
