# ML/AI Curriculum Glossary

This glossary defines key terms used across the deep reference guides. Terms are organized alphabetically for easy lookup. Each entry includes a simple definition, context, and cross-references to relevant week files.

## A
- **Activation Function**: A mathematical function applied to the output of a neuron in a neural network to introduce non-linearity. Examples: ReLU (Rectified Linear Unit), sigmoid. Helps the network learn complex patterns. (See: week17_18_deep_learning.md)
- **Actor-Critic**: A reinforcement learning method combining policy-based (actor) and value-based (critic) approaches. The actor chooses actions, the critic evaluates them. (See: week27_28_rl_advances.md)
- **Anomaly Detection**: Identifying data points that deviate significantly from normal behavior. Types: point, contextual, collective. (See: week9_10_anomaly_detection.md)
- **Autoencoder**: A neural network that learns to compress (encode) and reconstruct (decode) data. Used for dimensionality reduction or anomaly detection. (See: week9_10_anomaly_detection.md, week17_18_deep_learning.md)

## B
- **Backpropagation**: The algorithm used to train neural networks by calculating gradients and updating weights to minimize error. Works backwards from output to input. (See: week17_18_deep_learning.md)
- **Batch Normalization**: A technique to normalize inputs to each layer in a neural network, improving training stability and speed. (See: week17_18_deep_learning.md)
- **Brier Score**: A metric for evaluating probabilistic predictions. Measures the mean squared difference between predicted probabilities and actual outcomes. Lower is better. (See: week4_5_churn_prediction.md, week6_7_risk_scoring.md)

## C
- **Calibration**: Adjusting model outputs so predicted probabilities match real-world frequencies. Ensures a 70% predicted probability corresponds to 70% actual events. (See: week4_5_churn_prediction.md, week6_7_risk_scoring.md)
- **Churn Prediction**: Forecasting when customers will stop using a service. Involves binary classification with features like recency, frequency, monetary value. (See: week4_5_churn_prediction.md)
- **CNN (Convolutional Neural Network)**: A deep learning architecture for processing grid-like data (e.g., images). Uses convolutional layers to detect patterns like edges. (See: week17_18_deep_learning.md)
- **Cross-Validation**: A technique to evaluate model performance by splitting data into training and validation subsets multiple times. Prevents overfitting. (See: week1_data_exploration.md, week4_5_churn_prediction.md)

## D
- **Data Drift**: Changes in data distribution over time, causing model performance to degrade. Monitored using metrics like PSI (Population Stability Index). (See: week11_12_production_mlops.md)
- **Deep Learning**: A subset of machine learning using neural networks with many layers to learn complex representations. Requires large datasets and compute. (See: week17_18_deep_learning.md)
- **Dropout**: A regularization technique in neural networks where random neurons are "dropped" during training to prevent overfitting. (See: week17_18_deep_learning.md)

## E
- **EDA (Exploratory Data Analysis)**: The process of analyzing datasets to summarize main characteristics, often with visualizations. Includes checking distributions, correlations, missing values. (See: week1_data_exploration.md)
- **Embedding**: A dense vector representation of data (e.g., words or sentences) in a continuous space. Used in NLP for semantic similarity. (See: week33_39_nlp_semantic_search_llm.md)
- **Epoch**: One complete pass through the entire training dataset during neural network training. (See: week17_18_deep_learning.md)

## F
- **Feature Engineering**: Creating new features from raw data to improve model performance. Examples: log transformations, interaction terms, categorical encoding. (See: week1_data_exploration.md, week4_5_churn_prediction.md)
- **Fine-Tuning**: Adjusting a pre-trained model on a specific task with new data. Common in transfer learning. (See: week17_18_deep_learning.md, week33_39_nlp_semantic_search_llm.md)

## G
- **GLM (Generalized Linear Model)**: An extension of linear regression for different response types (e.g., binary outcomes with logistic regression). (See: week6_7_risk_scoring.md)
- **Gradient Descent**: An optimization algorithm to minimize loss by updating model parameters in the direction of the steepest descent. (See: week17_18_deep_learning.md)

## H
- **Homogeneity**: The assumption that residuals (errors) in a model have constant variance. Violated if errors increase with predictions. (See: week6_7_risk_scoring.md)
- **Hyperparameter Tuning**: Selecting the best settings for model parameters (e.g., learning rate) using techniques like grid search or Optuna. (See: week4_5_churn_prediction.md)

## I
- **Imbalance**: When classes in a dataset are unevenly distributed (e.g., 90% no-churn, 10% churn). Handled with SMOTE or class weights. (See: week1_data_exploration.md, week4_5_churn_prediction.md)
- **Isolation Forest**: An unsupervised algorithm for anomaly detection that isolates anomalies by randomly partitioning data. (See: week9_10_anomaly_detection.md)

## K
- **K-Fold Cross-Validation**: Dividing data into K subsets, training on K-1 and validating on the remaining one, repeating K times. (See: week1_data_exploration.md)

## L
- **Lasso Regression**: A linear model that adds L1 penalty to shrink coefficients, performing feature selection by setting some to zero. (See: week6_7_risk_scoring.md)
- **LLM (Large Language Model)**: AI models like GPT trained on vast text data for tasks like text generation and question answering. (See: week33_39_nlp_semantic_search_llm.md)
- **Local Outlier Factor (LOF)**: An anomaly detection method measuring local density deviation. Points with lower density than neighbors are anomalies. (See: week9_10_anomaly_detection.md)

## M
- **Markov Decision Process (MDP)**: A framework for reinforcement learning with states, actions, rewards, and transitions. (See: week27_28_rl_advances.md)
- **Multicollinearity**: When features are highly correlated, making it hard to determine individual effects. Detected with VIF. (See: week6_7_risk_scoring.md)

## N
- **Neural Network**: A model inspired by the brain, consisting of layers of interconnected nodes (neurons) that learn patterns from data. (See: week17_18_deep_learning.md)
- **NLP (Natural Language Processing)**: The field of AI dealing with understanding and generating human language. (See: week33_39_nlp_semantic_search_llm.md)

## O
- **One-Class SVM**: A support vector machine for anomaly detection trained on normal data to define a boundary. (See: week9_10_anomaly_detection.md)
- **Overfitting**: When a model performs well on training data but poorly on new data due to memorizing noise. Prevented with regularization. (See: week6_7_risk_scoring.md)

## P
- **Policy**: In reinforcement learning, a strategy for choosing actions based on states. Can be deterministic or stochastic. (See: week27_28_rl_advances.md)
- **Precision**: A metric for classification: proportion of positive predictions that are correct. (See: week4_5_churn_prediction.md)
- **Prompt Engineering**: Crafting input prompts to guide LLM responses effectively. (See: week33_39_nlp_semantic_search_llm.md)
- **PyTorch**: An open-source deep learning framework for building and training neural networks. (See: week17_18_deep_learning.md)

## Q
- **Q-Function**: In reinforcement learning, a value function estimating the expected reward for taking an action in a state. (See: week27_28_rl_advances.md)
- **Quantile Regression**: A regression technique predicting quantiles (e.g., median) instead of means. (See: week6_7_risk_scoring.md)

## R
- **R² (R-Squared)**: A metric measuring how much variance in the target is explained by the model. Ranges from 0 to 1. (See: week6_7_risk_scoring.md)
- **Recall**: A metric for classification: proportion of actual positives correctly identified. (See: week4_5_churn_prediction.md)
- **Reinforcement Learning (RL)**: A type of machine learning where agents learn by interacting with an environment to maximize rewards. (See: week27_28_rl_advances.md)
- **Residual**: The difference between predicted and actual values in regression. Analyzed for model diagnostics. (See: week6_7_risk_scoring.md)
- **Ridge Regression**: A linear model with L2 penalty to shrink coefficients and reduce overfitting. (See: week6_7_risk_scoring.md)
- **RNN (Recurrent Neural Network)**: A neural network for sequential data, maintaining memory of previous inputs. (See: week17_18_deep_learning.md)

## S
- **Semantic Search**: Searching based on meaning rather than exact keywords, using embeddings and vector similarity. (See: week33_39_nlp_semantic_search_llm.md)
- **SHAP (SHapley Additive exPlanations)**: A method to explain model predictions by attributing contributions to each feature. (See: week4_5_churn_prediction.md)
- **SMOTE (Synthetic Minority Oversampling Technique)**: A method to balance imbalanced datasets by generating synthetic examples for minority classes. (See: week1_data_exploration.md, week4_5_churn_prediction.md)

## T
- **Tokenization**: Splitting text into smaller units (tokens) like words or subwords for NLP processing. (See: week33_39_nlp_semantic_search_llm.md)
- **Transfer Learning**: Using a model trained on one task as a starting point for another related task. (See: week17_18_deep_learning.md)
- **Transformer**: A neural network architecture for handling sequential data, using attention mechanisms. Basis for models like BERT. (See: week33_39_nlp_semantic_search_llm.md)

## U
- **Underfitting**: When a model is too simple to capture patterns in the data, resulting in poor performance. (See: week6_7_risk_scoring.md)

## V
- **Validation Set**: A subset of data used to tune hyperparameters and evaluate model performance during training. (See: week1_data_exploration.md)
- **Variance Inflation Factor (VIF)**: A measure of multicollinearity; VIF > 5 indicates high correlation between features. (See: week6_7_risk_scoring.md)

## W
- **Weight**: In neural networks, parameters adjusted during training to control signal strength between neurons. (See: week17_18_deep_learning.md)

## Z
- **Z-Score**: A standardized score measuring how many standard deviations a value is from the mean. Used for outlier detection. (See: week1_data_exploration.md, week9_10_anomaly_detection.md)

---

## How to Use This Glossary
- **Lookup Terms:** Search alphabetically or by first letter.
- **Context:** Each definition includes where the term appears in the curriculum.
- **Further Reading:** Cross-references link to specific week files for deeper explanations.
- **Updates:** Add new terms as you progress through weeks.

If a term is missing, check the relevant week file or ask for clarification!