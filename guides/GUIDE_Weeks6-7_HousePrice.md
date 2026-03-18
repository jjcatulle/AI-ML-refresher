# Guide: House Price Prediction (Weeks 6-7)

## Big Picture
Build a regression model to predict house prices based on features like size, location, bedrooms, and age.

**Why?** Bridges classification (Churn) → Regression (continuous values), introduces feature engineering and model comparison.

**Key Skills:**
- Feature engineering (scaling, encoding, creating new features)
- Regression metrics (MAE, RMSE, R²)
- Model comparison (Linear Regression vs Random Forest vs Gradient Boosting)
- Cross-validation to prevent overfitting

---

## Concept 1: The Regression Problem

**What:** Predicting continuous values (prices, salaries, temperatures) vs categories.

**Example:**
```python
# Classification (Churn)
pred = 0  # Not churning
pred = 1  # Churning

# Regression (Price)
pred = 250000.50  # Any continuous value
pred = 450000.75
```

**Why Different?** Use different metrics:
- **Classification:** Accuracy, Precision, Recall, ROC-AUC
- **Regression:** MAE, RMSE, R² (correlation between predicted & actual)

---

## Concept 2: Feature Engineering Fundamentals

**What:** Creating/transforming features to improve model performance.

```python
# Raw feature
age = 25
bedrooms = 3
square_feet = 2500

# Engineered features
age_squared = age ** 2
price_per_sqft = price / square_feet
is_old = 1 if age > 50 else 0
```

**Types:**
1. **Scaling** - Normalize to 0-1 range (StandardScaler)
2. **Encoding** - Convert categories to numbers
3. **Creation** - Combine features (ratios, polynomials)
4. **Selection** - Keep only important features

**When to Use:**
- Always scale for: Linear Regression, KNN, neural networks
- Don't need for: Tree-based (Random Forest, XGBoost)

---

## Concept 3: Handling Categorical Features

**What:** Converting text categories to numbers.

```python
# Raw
location = ['Urban', 'Suburban', 'Rural']
house_type = ['Apartment', 'House', 'Condo']

# One-hot encoding
location_Urban = [1, 0, 0]
location_Suburban = [0, 1, 0]
location_Rural = [0, 0, 1]

# Using pandas
pd.get_dummies(df['location'], prefix='loc')
# Result: loc_Urban, loc_Suburban, loc_Rural columns
```

**When to Use:**
- Tree-based models: Direct encoding (0, 1, 2) works
- Linear/Neural: One-hot encoding to avoid false ordering

---

## Concept 4: Train/Test/Validation Split

**What:** Ensuring model generalizes to unseen data.

```python
from sklearn.model_selection import train_test_split

# 70% train, 20% test, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.67, random_state=42
)

# Typical sizes: 70/20/10 or 80/20 (train/test)
```

**Why Three Sets?**
- **Train:** Model learns here
- **Validation:** Tune hyperparameters here
- **Test:** Final evaluation (touch only once!)

---

## Concept 5: Cross-Validation

**What:** Multiple train/test splits to get robust performance estimate.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, cv=5, scoring='r2'
)
# 5 folds: 80% train, 20% test (repeated 5 times)
print(f"Scores: {scores}")
print(f"Average: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")  # Lower is better (consistent)
```

**When to Use:**
- Small datasets (< 10k samples)
- Final model evaluation
- Hyperparameter tuning

---

## Concept 6: Regression Model Types

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Get coefficients (feature importance)
for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature}: {coef:.4f}")
```

**Pros:** Fast, interpretable  
**Cons:** Assumes linear relationships

### Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Feature importance
importances = model.feature_importances_
```

**Pros:** Handles non-linear relationships, robust  
**Cons:** Less interpretable, slower

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Pros:** Often best performance  
**Cons:** Slower, hyperparameter tuning needed

---

## Concept 7: Regression Metrics

### MAE (Mean Absolute Error)
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
# Average error in dollars
# MAE = 25000 means average prediction is off by $25,000
```

### RMSE (Root Mean Squared Error)
```python
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, predictions))
# Penalizes large errors more than MAE
# RMSE = 35000 when one prediction is very wrong
```

### R² Score
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
# 1.0 = perfect, 0.0 = baseline, negative = worse than baseline
# Interpretation: R² = 0.85 means model explains 85% of variance
```

**When to Use:**
- **MAE:** Easy interpretation (average dollars off)
- **RMSE:** Penalize large errors more
- **R²:** Understand total model quality

---

## Concept 8: Feature Importance & Analysis

```python
# Random Forest
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices[:10]])
plt.xlabel('Importance')
plt.title('Top 10 Most Important Features')
plt.show()
```

**Interpretation:**
- High importance = strong predictor
- Low importance = consider removing
- Used for business insights (what drives prices?)

---

## Concept 9: Residual Analysis

**What:** Analyzing prediction errors to detect problems.

```python
# Calculate residuals
residuals = y_test - predictions

# Plot
plt.figure(figsize=(12, 4))

# Residuals vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

# Distribution of residuals
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.tight_layout()
plt.show()

# Should look like random noise around 0
```

**Red Flags:**
- Residuals trend upward/downward = Model has bias
- Residuals fan out = Predictions worse at extremes
- Residuals not normal = Try different model

---

## Concept 10: Hyperparameter Tuning

**What:** Finding best settings for model.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
```

**Common Hyperparameters:**
- **Random Forest:** n_estimators, max_depth, min_samples_split
- **Gradient Boost:** learning_rate, n_estimators, max_depth
- **Linear:** regularization (L1/L2)

---

## Challenge Approach

### Challenge 1-3: Data Exploration & Cleaning
- Load California Housing or similar dataset
- Check shape, data types, missing values
- Summary statistics for each feature
- Visualize distributions and correlations

### Challenge 4-6: Feature Engineering
- Scale numerical features (StandardScaler)
- Encode categorical features (one-hot)
- Create new features (ratios, polynomials)
- Remove highly correlated features

### Challenge 7-9: Model Training & Evaluation
- Train 3 models (Linear, Random Forest, Gradient Boost)
- Use cross-validation for robust scoring
- Calculate MAE, RMSE, R² for each
- Make predictions on test set

### Challenge 10-12: Analysis & Interpretation
- Compare model performance
- Extract and visualize feature importance
- Analyze residuals to find issues
- Write business summary (what drives prices?)

---

## Key Takeaways

✅ **Regression = predicting continuous values** (uses different metrics than classification)

✅ **Feature engineering matters** (scaled/engineered features often beat raw data)

✅ **Cross-validation prevents overfitting** (use for final model evaluation)

✅ **Multiple models comparison** (Linear, Ensemble, Boosting each has strengths)

✅ **Residual analysis catches problems** (random noise around 0 = good model)

✅ **Feature importance = business insights** (explains what drives outcomes)
