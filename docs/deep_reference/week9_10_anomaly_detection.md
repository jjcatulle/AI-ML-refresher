# Weeks 9-10: Anomaly Detection (Deep Dive)

## 1. Core Concepts
- Anomaly types:
  - Point anomalies: one sample is abnormal (fraud, sensor spike).
  - Contextual anomalies: normal region depends on context (time series seasonality).
  - Collective anomalies: group is anomalous despite individuals looking normal (network intrusions).
- Data assumptions:
  - Gaussian (z-score), heavy-tailed, mixed-type.
  - Temporal dependency, spatial correlation.
- Goal: detect rare events with high recall + interpretability.

## 2. Analytical Setup
- Define business objectives: alert volume target, precision floor, mean time to detect.
- Preprocessing pipeline:
  - Imputation: KNN, interpolation.
  - Scaling: StandardScaler, RobustScaler.
  - Feature aggregation: roll/sum/expanding + differencing.
  - Feature augmentation: FFT, wavelets, entropy, moving averages, seasonal components.
- Anomaly score calibration:
  - Convert raw model output to score in [0,1] using percentile or logistic transform.

## 3. Algorithms & Complexity
### 3.1 Statistical methods
- Z-score: `|x - mu| / sigma > tau`. Works for univariate and stationarized series.
- IQR rule: `x < Q1-1.5*IQR or x > Q3 + 1.5*IQR`.
- Gaussian Mixture Model (GMM): log-likelihood below threshold.

### 3.2 Distance/density methods
- k-NN: anomaly if average distance to k nearest neighbors is large.
- Local Outlier Factor (LOF): ratio of local density.
- DBSCAN: detect small clusters/noise points.

### 3.3 Tree-based and ensembles
- Isolation Forest: random partitioning, path length heuristic (lower path = anomaly).
- One-Class SVM: learns boundary in feature space; parameter `nu` for expected anomaly fraction.

### 3.4 Representation + Deep methods
- Autoencoder (AE): reconstruct input, anomaly if reconstruction error > threshold.
- Variational AE: includes latent distribution for better anomaly score distribution.
- LSTM AE for sequences: encodes time window, uses MSE on reconstruction.

## 4. Time Series specifics
- Windowing:
  - Sliding window: predict anomaly for center point using past N points.
  - Expanding window: cumulative train.
- Detrending and deseasonalization using STL/Pandas.
- Residual-based detection: predict with ARIMA/Prophet, anomaly if residual beyond k*sigma.
- Multivariate seasonal anomaly: Matrix Profile, HOTSAX, NAD.

## 5. Evaluation and metrics
- Confusion matrix with small positives:
  - precision, recall, F1, FPR.
  - precision@k and recall@k for top-k alerts.
- ROC-AUC / PR-AUC with pseudo labels, if true labels unavailable use proxy eval on synthetic injection.
- Time-based error metrics:
  - Detection delay (time from anomaly start to first alert)
  - Alert rate per time period.

## 6. Deployment patterns
- Offline → online architecture:
  - Build model on full dataset, export pipeline (scaling + features + model) as `joblib`.
  - Stream inference: process event windows in batch, maintain state in DB.
- Evolution strategy:
  - Use sliding retrain schedule (daily/weekly) with periodic drift detection (KL divergence, PSI).
  - Auto-thresholding: quantile-based thresholds updated on recent normal data.
- Alert/Dashboard:
  - Expose anomaly scores, anomaly category, provenance features.
  - Include context variables: expected range, past behavior, confidence.

## 7. Tools and code snippets
- `pyod` library quick start:
  - `from pyod.models.iforest import IForest`
  - `clf = IForest(contamination=0.01); clf.fit(X_train); scores = clf.decision_function(X_test)`
- Autoencoder in PyTorch example (simplified):
  - `loss = criterion(reconstructed, x)` with early stopping + rolling threshold.
- `river` for streaming detection:
  - `from river.anomaly import HSTree`; `model = HSTree()`; incremental `model.learn_one(x); score = model.score_one(x)`.

## 8. Research and references
- Papers:
  - Liu et al., "Isolation Forest" (2008).
  - Chandola, Banerjee, Kumar, "Anomaly Detection: A Survey" (2009).
  - Laptev et al., "Generic and Scalable Framework for Automated Time Series Anomaly Detection".
- Libraries:
  - `pyod`, `scikit-multiflow`, `river`, `tensorflow-io`.
- Tutorials:
  - Kaggle: "Detecting Anomalies in Time Series".
  - Towards Data Science: "Guide to Anomaly Detection with Python".

## 9. Deep challenge
- Build a complete pipeline for multivariate time-series anomaly detection:
  1. Ingest raw events, compute rolling features + seasonal decomposition.
  2. Train a hybrid model: LOF + LSTM-Autoencoder.
  3. Generate a combined score: `score = 0.6*norm_lof + 0.4*recon_error`.
  4. Validate with anomaly-injection test, compute precision@50 and detection delay.
  5. Serve via FastAPI endpoint that returns `anomaly_score`, `threshold`, `is_anomaly`, and `explanation`.

