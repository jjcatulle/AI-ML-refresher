# Weeks 4-5: Churn Prediction Model (Deep Dive)

## 1. Problem Framing
- Define unit of churn: account cancellation, inactivity window, subscription lapse.
- KPI alignment:
  - business: cost of losing a customer, customer lifetime value (CLV).
  - model: recall@k, lift, cost-sensitive accuracy.
- Churn horizon: 30/60/90 days and prediction lead time.
- Baseline:
  - class prior rule (predict no churn) and business rule (if last purchase > 60 days then churn).
  - compare to logistic regression with pseudo-probabilities.

## 2. Data Preparation (deep)
- Data quality checks:
  - account age vs churn label consistency.
  - ghost accounts (no activity) handling.
- Label definition and leakage:
  - use a sliding window; ensure features are computed from past data only.
  - avoid future leaks from `days_since_last_purchase` when close to prediction date.
- Feature categories:
  - Behavioral: recency/frequency/monetary (RFM), session length, churn-susceptibility flags.
  - Product interaction: categories used, churn offers seen.
  - Support activity: tickets, NPS scores.
- Feature transformations:
  - `log1p` for skewed monetary values.
  - `rank` and percentile features for heavy-tailed distribution.

## 3. Feature engineering and selection
- Automatic selectors:
  - `sklearn.feature_selection.RFECV(estimator=LogisticRegression())`.
  - `SelectFromModel` using tree importance.
- Statistical tests:
  - ANOVA/t-test for numeric and chi-squared for categorical.
- Correlation and multicollinearity:
  - drop features with `VIF > 5`.

## 4. Model building
- Candidate models:
  - `LogisticRegression(C=1.0, penalty='l1', solver='saga', class_weight='balanced')`.
  - `RandomForestClassifier(n_estimators=400, max_depth=14, class_weight='balanced')`.
  - `xgboost.XGBClassifier(scale_pos_weight=pos_weight)`.
- Tuning with Optuna example:
  ```python
  import optuna
  def objective(trial):
      params = {
          'max_depth': trial.suggest_int('max_depth', 2, 10),
          'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
          'subsample': trial.suggest_float('subsample', 0.5, 1.0),
      }
      clf = xgboost.XGBClassifier(**params)
      score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc').mean()
      return score
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=50)
  ```

## 5. Imbalance strategy
- Create a pipeline:
  - `SMOTE` + `RandomForestClassifier`
  - `CalibratedClassifierCV` for probability calibration.
- When using class weight:
  - `class_weight={'0': 1, '1': weight}` where `weight = n_samples/ (2*class_count)`.
- Evaluate with F1 score and precision@k to manage false positives.

## 6. Explainability + trust
- SHAP:
  - global importance, force plot for single prediction.
  - `shap.TreeExplainer(model).shap_values(X)`.
- Partial dependence: `sklearn.inspection.plot_partial_dependence`.
- Model fairness:
  - demographic parity, equality of odds across segments.

## 7. Validation strategy
- Time-series split (rolling prequential):
  - Train on months T-6..T-2, validate on T-1.
- Backtest with cohort analysis.
- Calibration checks:
  - `CalibrationDisplay.from_estimator(model, X_valid, y_valid, n_bins=10)`.
- Brier score: `brier_score_loss(y_valid, y_prob)`.

## 8. Deployment & monitoring
- Build a `Pipeline`:
  ```python
  pipe = Pipeline([('preproc', preprocess), ('model', clf)])
  joblib.dump(pipe, 'churn_model.joblib')
  ```
- Online scoring:
  - postgres pipeline inserts features; microservice computes churn probability.
- Drift detection:
  - KS test on numeric predictors.
  - Population Stability Index (PSI) per feature.

## 9. References
- Kaggle notebook: "Telco Churn" with XGBoost baseline.
- Paper: "Addressing Data Imbalance for Churn Predictive Modeling".
- GitHub: `sf-knn` articles for churn interpretable metrics.

## 10. Challenge
- Add a decayed event feature:
  - `decay_value = sum(ev * exp(-lambda * time_delta))`.
- Run variant experiments with/without this feature and evaluate gain in PR-AUC.
- Build a dashboard with:
  - distribution of churn scores, risk segment, top SHAP features by segment.

