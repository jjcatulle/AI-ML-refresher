# 📋 Quick Reference: ML/Data Science Cheat Sheet

Keep this handy while you code!

---

## 🐼 Pandas Essentials

### Loading & Saving
```python
import pandas as pd

df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx')
```

### Exploration
```python
df.head()                    # First 5 rows
df.tail(10)                  # Last 10 rows
df.shape                     # (rows, columns)
df.dtypes                    # Data types
df.info()                    # Full summary
df.describe()                # Statistics
df.columns                   # Column names
```

### Cleaning
```python
# Missing values
df.isnull().sum()            # Count NaN
df.dropna()                  # Remove NaN rows
df.fillna(0)                 # Fill with 0
df.fillna(df.mean())         # Fill with mean

# Duplicates
df.duplicated().sum()        # Count duplicates
df.drop_duplicates()         # Remove duplicates

# Data types
df['col'] = df['col'].astype(float)
```

### Selection & Filtering
```python
df['col_name']               # Select column
df[['col1', 'col2']]         # Multiple columns
df.loc[0]                    # Row by label
df.iloc[0]                   # Row by position
df[df['age'] > 18]           # Filter
df[(df['age'] > 18) & (df['salary'] > 50000)]  # Multiple filters
```

### Grouping & Aggregation
```python
df.groupby('category').sum()
df.groupby('category').mean()
df.groupby(['cat1', 'cat2']).agg({'salary': 'mean', 'age': 'max'})
df['col'].value_counts()     # Count occurrences
df['col'].nunique()          # Unique values
```

---

## 🔢 NumPy Essentials

```python
import numpy as np

# Arrays
arr = np.array([1, 2, 3])
arr = np.zeros(10)           # Array of zeros
arr = np.ones(10)            # Array of ones
arr = np.arange(0, 10, 2)    # 0, 2, 4, 6, 8
arr = np.linspace(0, 10, 5)  # 5 equal spaces 0-10

# Operations
arr + 5                      # Add to all
arr * 2                      # Multiply all
np.sqrt(arr)                 # Square root
np.mean(arr)                 # Average
np.std(arr)                  # Standard deviation
np.max(arr)                  # Maximum
```

---

## 📊 Matplotlib & Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot
plt.plot(x, y)
plt.show()

# Histogram
plt.hist(data, bins=20)

# Scatter plot
plt.scatter(x, y)

# Box plot
plt.boxplot([data1, data2])

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x)
axes[0, 1].hist(y)

# Seaborn
sns.heatmap(correlation_matrix, annot=True)
sns.boxplot(data=df, x='category', y='value')
sns.scatterplot(data=df, x='x', y='y', hue='category')
```

---

## 🤖 Scikit-learn Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 🎯 Classification Models

### Logistic Regression (Simple)
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
```

### Decision Tree (Interpretable)
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)
```

### Random Forest (Often Best)
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### XGBoost (Advanced)
```python
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 📈 Regression Models

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### Ridge Regression (Prevents overfitting)
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

### Random Forest Regression
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 📊 Evaluation Metrics

### Classification
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)          # Overall correctness
precision = precision_score(y_test, y_pred)        # True positives / predicted positives
recall = recall_score(y_test, y_pred)              # True positives / actual positives
f1 = f1_score(y_test, y_pred)                      # Harmonic mean of precision & recall

cm = confusion_matrix(y_test, y_pred)              # True/False positives/negatives
print(classification_report(y_test, y_pred))
```

### Regression
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)           # Average squared error
rmse = np.sqrt(mse)                                # Root mean squared error
mae = mean_absolute_error(y_test, y_pred)         # Average absolute error
r2 = r2_score(y_test, y_pred)                      # Explained variance (0-1)
```

---

## 🔄 Cross-Validation & Hyperparameter Tuning

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
```

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

## 🧠 Deep Learning (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Setup
model = NeuralNet(input_size=10, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
```

---

## 🤗 LangChain Essentials

```python
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Create model
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Explain {topic} in simple terms."
)

# Create chain
chain = prompt | llm

# Run
result = chain.invoke({"topic": "machine learning"})
print(result.content)
```

### RAG (Retrieval-Augmented Generation)
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA

# 1. Load documents
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Store in vector DB
vector_store = Pinecone.from_documents(chunks, embeddings, index_name="my-index")

# 5. Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# 6. Query
answer = qa_chain.invoke({"query": "What is the document about?"})
```

---

## 🐳 FastAPI Model Serving

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="ML Model API")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define request/response schemas
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

# Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    features = [request.feature1, request.feature2, request.feature3]
    prediction = model.predict([features])[0]
    confidence = 0.95
    return PredictionResponse(prediction=prediction, confidence=confidence)

@app.get("/health")
async def health():
    return {"status": "ok"}

# Run with: uvicorn main:app --reload
```

---

## 🐳 Docker Quick Start

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Building & Running
```bash
# Build image
docker build -t my-ml-app .

# Run container
docker run -p 8000:8000 my-ml-app

# Run with volume mount (for development)
docker run -v $(pwd):/app -p 8000:8000 my-ml-app
```

---

## 📌 Common Patterns

### Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Scale Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Handle Missing Values
```python
df = df.dropna()  # Remove rows with NaN
# OR
df = df.fillna(df.mean())  # Fill with mean
```

### Encode Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
```

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
```

---

## ⏱️ Debugging Tips

### Check for Overfitting
- High training accuracy, low test accuracy
- Solution: Use regularization, more data, simpler model

### Check for Underfitting
- Low both training and test accuracy
- Solution: More features, more complex model, more training

### Class Imbalance
- Minority class predictions are bad
- Solution: Use `class_weight='balanced'` or stratified split

### Scaling Issues
- Model performance is bad
- Solution: Scale features with StandardScaler or MinMaxScaler

### Convergence Issues
- Model takes too long to train
- Solution: Lower learning rate, smaller batch size, normalize data

---

## 🎓 When Stuck

1. **Read the error message carefully** - usually tells you exactly what's wrong
2. **Google the error** + library name
3. **Check official documentation**
4. **Try a simpler version** of your code
5. **Ask on Stack Overflow** with reproducible example

---

## 📚 Keep This File Open While Coding!

Print it, bookmark it, or keep it in your IDE. Reference it constantly!

---

*Last Updated: March 18, 2026*
