# Hyperparameter Tuning Guide

Use this when your baseline works, but you need to improve it in a controlled way.

## What tuning is
Hyperparameter tuning means choosing better model or system settings based on evidence from a validation process.

Examples:
- Logistic Regression: `C`, `penalty`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
- XGBoost: `learning_rate`, `max_depth`, `subsample`
- Neural networks: learning rate, batch size, dropout, epochs
- RAG systems: chunk size, overlap, `top_k`, reranker settings, prompt instructions
- Agents: model choice, tool selection rules, retries, timeouts, guardrail thresholds

Tuning is not:
- changing random things until the score looks better
- using the test set again and again to make decisions
- skipping baselines and jumping straight to large searches

## Why tuning matters
A model can underperform for three very different reasons:
1. The data is weak or leaky.
2. The features or retrieval setup are weak.
3. The model or system settings are poor.

Tuning only helps with the third problem.
If the real issue is bad data or leakage, tuning can waste time and give fake confidence.

## What a hyperparameter is
A hyperparameter is a setting chosen before or during training that shapes model behavior.

Examples by model family:

### Linear models
- `C` in Logistic Regression: lower values mean stronger regularization
- `alpha` in Ridge/Lasso: larger values shrink coefficients more strongly

### Tree models
- `max_depth`: limits how deep trees can grow
- `n_estimators`: number of trees in the ensemble
- `min_samples_split`: minimum samples needed to split a node

### Boosting models
- `learning_rate`: how aggressively each round updates
- `n_estimators`: how many boosting rounds are used
- `max_depth`: tree complexity inside each boosting step

### Neural networks
- learning rate
- batch size
- hidden size / number of layers
- dropout
- weight decay
- number of epochs

### RAG and retrieval systems
- chunk size and chunk overlap
- `top_k`
- similarity threshold
- reranker usage and cutoff
- prompt structure

## What to tune first
Do not tune everything at once.

Use this order:
1. Fix data quality and leakage first.
2. Build a valid baseline.
3. Pick the metric that matches the real goal.
4. Identify the failure mode.
5. Tune the small number of settings most likely to affect that failure mode.

Examples:
- If a churn model misses too many churners, tune threshold and class weighting before trying ten different models.
- If a house price model makes a few huge misses, tune regularization or tree depth before adding more complexity.
- If a RAG bot misses the right documents, tune chunking and retrieval settings before rewriting the prompt.

## The correct tuning workflow

### 1) Start with a baseline
Always begin with a simple baseline model or system.

Examples:
- tabular classification: Logistic Regression
- tabular regression: Linear Regression or Ridge
- text classification: TF-IDF + Logistic Regression
- RAG: BM25 or embeddings-only baseline
- image classification: pretrained backbone with default head

### 2) Hold out a test set
Use the test set only for final reporting.
Do not repeatedly tune against the test set.

Typical structure:
- training set: fit model
- validation set or cross-validation: choose settings
- test set: final honest evaluation

### 3) Pick a search strategy

#### Manual tuning
Use when:
- you are learning
- you have one or two parameters
- you want to understand cause and effect

Example:
- try `max_depth` values 3, 5, 8
- compare recall and precision

#### Grid search
Use when:
- search space is small
- you know exactly which values are worth testing

Pros:
- simple
- reproducible

Cons:
- expensive if too many combinations
- wastes time on bad regions of the search space

#### Random search
Use when:
- there are many parameters
- you want a better exploration-to-cost tradeoff

Pros:
- usually better than grid search for larger spaces
- finds strong regions faster

#### Bayesian / Optuna-style search
Use when:
- search is expensive
- model training takes time
- you want smarter parameter proposals

Pros:
- efficient for expensive tuning
- often reaches better results with fewer trials

Cons:
- harder to explain than manual/grid search when you are still learning

## Cross-validation and why it matters
Cross-validation gives a more reliable estimate than one random split.

Typical idea:
- divide training data into several folds
- train on most folds
- validate on the remaining fold
- repeat and average scores

Why this matters:
- reduces luck from one split
- gives more stable comparison between parameter choices
- is especially useful on smaller datasets

## The most common tuning mistake
The most common tuning mistake is optimizing for the wrong thing.

Examples:
- tuning for accuracy on imbalanced fraud data
- tuning ROC-AUC when the real business action depends on recall at a chosen threshold
- tuning only model settings when the real issue is missing-value handling or leakage

## Tuning by project type

### Tabular classification
Good early parameters to tune:
- threshold
- `class_weight`
- `max_depth`
- `n_estimators`
- `min_samples_leaf`
- regularization strength like `C`

Example goal:
- improve recall without destroying precision

Suggested order:
1. threshold
2. class balancing
3. tree depth / regularization
4. ensemble size

### Tabular regression
Good early parameters to tune:
- `alpha` for Ridge/Lasso
- `max_depth`
- `n_estimators`
- `learning_rate`

Goal:
- reduce MAE or RMSE, especially large misses

### Imbalanced classification
Good early parameters to tune:
- class weights
- threshold
- sampling strategy
- metric choice

Goal:
- improve minority-class detection honestly

### RAG systems
Good early parameters to tune:
- chunk size
- chunk overlap
- `top_k`
- reranker cutoff
- prompt structure
- retrieval filters

Goal:
- first improve document retrieval quality, then answer quality

Do not start by tuning everything.
If the right documents are not retrieved, prompt tuning will not solve the root problem.

### Neural nets / CNNs
Good early parameters to tune:
- learning rate
- batch size
- epochs
- dropout
- augmentation strength
- frozen vs unfrozen layers

Goal:
- improve validation performance without overfitting

Watch for:
- validation loss rising while training loss falls
- unstable training caused by poor learning rate choices

## Threshold tuning is related but different
Hyperparameter tuning changes system settings.
Threshold tuning changes the decision boundary after scoring.

Example:
- predicted churn probability is `0.73`
- default threshold is `0.50`
- business may prefer `0.35` if recall is more important

This matters because the best model score is not always the best action threshold.

## Simple examples

### Example 1: Grid search for Random Forest
Why this example matters:
- Use this when you have a small, believable search space.
- `scoring='f1'` makes the search optimize the behavior you actually care about instead of default accuracy.
- `cv=5` reduces luck from one split.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
}

search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
)

search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)
```

How to read it:
- If `best_score_` is only slightly better than baseline, the tuning gain may not be meaningful.
- If the best model becomes much more complex for a tiny gain, it may not be the right production choice.

### Example 2: Randomized search
Why this example matters:
- Use this when there are too many combinations for grid search to be efficient.
- `n_iter=12` lets you explore more broadly at lower cost.
- This is often a better default than grid search once the search space gets larger.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=12,
    scoring='f1',
    cv=5,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)
print(search.best_params_)
```

How to read it:
- Randomized search is useful when you do not know the best region yet.
- If the same kinds of values keep winning, narrow the search around that region next.

### Example 3: Threshold tuning
Why this example matters:
- The model may already rank cases well, but the default `0.50` threshold may be wrong for your business goal.
- In churn or fraud work, threshold tuning often improves behavior faster than changing the entire model.

```python
import numpy as np
from sklearn.metrics import recall_score, precision_score

proba = model.predict_proba(X_valid)[:, 1]

for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (proba >= t).astype(int)
    print(
        t,
        'precision=', round(precision_score(y_valid, pred, zero_division=0), 3),
        'recall=', round(recall_score(y_valid, pred), 3),
    )
```

How to read it:
- Lower thresholds usually increase recall and reduce precision.
- Higher thresholds usually increase precision and reduce recall.
- Choose the threshold that matches the real action, not the prettiest single metric.

### Example 4: Optuna for smarter search
Why this example matters:
- Use this when the search space is larger and each training run is expensive.
- Optuna proposes new trials based on previous results, so it often finds good regions faster than manual or grid search.

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int('n_estimators', 100, 500),
        max_depth=trial.suggest_int('max_depth', 3, 20),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

print(study.best_params)
print(study.best_value)
```

How to read it:
- If the best parameters are repeatedly near the search boundary, your search range may be too narrow.
- If the best value is only a tiny lift over baseline, more tuning may not be the highest-value next step.

### Example 5: Tuning evidence log in code
Why this example matters:
- Tuning is much easier to trust when each run is logged in a consistent structure.
- This helps you compare baseline vs tuned runs without relying on memory.

```python
import pandas as pd

tuning_runs = [
    {
        'run_name': 'baseline_logreg',
        'setting_changed': 'none',
        'metric_optimized': 'f1',
        'valid_score': 0.61,
        'notes': 'baseline before threshold tuning',
    },
    {
        'run_name': 'threshold_0_40',
        'setting_changed': 'threshold=0.40',
        'metric_optimized': 'recall',
        'valid_score': 0.78,
        'notes': 'recall improved, precision dropped',
    },
]

tuning_log_df = pd.DataFrame(tuning_runs)
tuning_log_df
```

How to read it:
- Good tuning logs capture both the score and the tradeoff.
- The best run is not always the one with the highest score if it causes unacceptable side effects.

## How to read tuning results
Do not only ask: "Which trial has the highest score?"
Also ask:
- Is the improvement meaningful or tiny?
- Is the improvement stable across folds?
- Did precision, recall, latency, or cost get worse?
- Does the best setting still make sense for production?

## What to record every time you tune
Write down:
- baseline model/system
- dataset split used
- metric optimized
- parameters tested
- best setting found
- validation score
- final test score
- what changed in behavior

## Tuning report template

Use this at the end of every project notebook.

```text
Tuning target:
Baseline system/model:
Baseline metric(s):

Failure mode I was trying to improve:

Setting(s) tuned:
Why I chose these first:

Validation method used:
Metric optimized:

Best setting found:
Best validation result:

What improved:
What got worse:
Tradeoff accepted:

Final test result:
Next tuning step if I had more time:
```

## Project-specific tuning report variants

### Churn / tabular classification
```text
Tuning target: improve churn detection quality
Baseline model:
Baseline recall / precision / F1:

Failure mode I was trying to improve:
- missing likely churners
- too many false positives
- unstable performance across customer segments

Setting(s) tuned:
- threshold
- class_weight
- C / max_depth / n_estimators / min_samples_leaf

Why I chose these first:

Validation method used:
Metric optimized:

Best setting found:
Best validation result:

What improved:
What got worse:
Segment or slice still weak:

Final test result:
Business tradeoff accepted:
Next tuning step:
```

### House price / regression
```text
Tuning target: reduce prediction error
Baseline model:
Baseline MAE / RMSE / R^2:

Failure mode I was trying to improve:
- large misses on expensive homes
- underfitting overall
- unstable errors across neighborhoods or home sizes

Setting(s) tuned:
- alpha
- max_depth
- n_estimators
- learning_rate

Why I chose these first:

Validation method used:
Metric optimized:

Best setting found:
Best validation result:

What improved:
What got worse:
Largest remaining error cases:

Final test result:
Tradeoff accepted:
Next tuning step:
```

### Fraud / imbalanced classification
```text
Tuning target: improve minority-class detection
Baseline model:
Baseline precision / recall / PR-AUC / F1:

Failure mode I was trying to improve:
- too many missed fraud cases
- too many false alerts
- weak performance at the operating threshold

Setting(s) tuned:
- threshold
- class_weight
- sampling strategy
- max_depth / learning_rate / n_estimators

Why I chose these first:

Validation method used:
Metric optimized:

Best setting found:
Best validation result:

What improved:
What got worse:
Operational cost of new false positives:

Final test result:
Tradeoff accepted:
Next tuning step:
```

### RAG / retrieval systems
```text
Tuning target: improve retrieval and grounded answer quality
Baseline system:
Baseline recall@k / precision@k / groundedness:

Failure mode I was trying to improve:
- wrong documents retrieved
- correct documents retrieved but answer still weak
- answer not grounded in sources

Setting(s) tuned:
- chunk size / overlap
- top_k
- reranker cutoff
- prompt structure
- retrieval filters

Why I chose these first:

Validation method used:
Metric optimized:

Best setting found:
Best validation result:

What improved:
What got worse:
Remaining retrieval weakness:

Final test result:
Tradeoff accepted:
Next tuning step:
```

### Neural networks / CNNs
```text
Tuning target: improve validation performance without overfitting
Baseline model:
Baseline validation metric:

Failure mode I was trying to improve:
- validation plateau
- overfitting
- unstable training

Setting(s) tuned:
- learning rate
- batch size
- epochs
- dropout
- augmentation
- frozen vs unfrozen layers

Why I chose these first:

Validation method used:
Metric optimized:

Best setting found:
Best validation result:

What improved:
What got worse:
Overfitting / underfitting signal now:

Final test result:
Tradeoff accepted:
Next tuning step:
```

### Agents / production systems / capstone
```text
Tuning target: improve full-system behavior
Baseline system:
Baseline success / latency / cost / reliability:

Failure mode I was trying to improve:
- poor task success
- too many retries or tool failures
- latency too high
- quality improved but cost too high

Setting(s) tuned:
- prompt structure
- tool rules
- retries / timeouts
- retrieval settings
- thresholds / caching / batching

Why I chose these first:

Validation method used:
Metric optimized:

Best setting found:
Best validation result:

What improved:
What got worse:
Production tradeoff accepted:

Final test result:
Next tuning step:
```

## Tuning checklist
Before tuning:
- data cleaned
- leakage checked
- baseline exists
- metric chosen
- test set protected

During tuning:
- change one meaningful group of settings at a time
- use validation or cross-validation
- keep notes for every run

After tuning:
- compare against baseline
- check slices and failure cases
- run final test once
- document tradeoffs

## Where this should show up in your projects
You should be able to answer these questions in every project notebook:
1. What baseline did I start with?
2. What parameter or threshold did I tune first, and why?
3. What metric did I optimize?
4. Did tuning improve the right behavior, or only one number?
5. What would I tune next if I had one more day?

## References
Internal references:
- docs/QUICK_REFERENCE.md
- docs/ONE_PAGE_ML_FLOW_CARD.md
- docs/FULL_ML_FLOW_CHECKLIST.md
- docs/EVALUATION_FRAMEWORK.md
- docs/AI-ML-LEARNING-GUIDE.md

Official references:
- https://scikit-learn.org/stable/modules/grid_search.html
- https://scikit-learn.org/stable/modules/cross_validation.html
- https://scikit-learn.org/stable/model_selection.html
- https://scikit-learn.org/stable/common_pitfalls.html
- https://optuna.org/
