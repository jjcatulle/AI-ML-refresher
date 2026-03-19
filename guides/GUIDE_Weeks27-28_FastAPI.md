# Guide: FastAPI Deployment (Weeks 27-28)

## Big Picture
Deploy your ML models as REST APIs using FastAPI, containerization, and cloud platforms.

**Why?** ML models in notebooks don't help users. APIs make models accessible.

**Key Skills:**
- FastAPI framework and routing
- Request/response validation
- Model loading and inference
- Error handling and logging
- Deployment and scaling

## 💼 Real-World Use Cases
- **Productization:** Serve model predictions to web/mobile apps.
- **Internal tools:** Provide APIs for data teams to access ML models easily.
- **Automated reporting:** Trigger predictions on a schedule (cron) and store results.

---

## 🎯 Recommended Models for Weeks 27-28

For FastAPI, you need a **pre-trained model** (not a dataset). Choose ONE:

### Option 1: Iris Classifier ✅ **EASIEST & RECOMMENDED**
- **What:** Pre-trained model that classifies iris flowers
- **How to create:**
  ```python
  from sklearn.datasets import load_iris
  from sklearn.ensemble import RandomForestClassifier
  import joblib
  
  # Train once
  iris = load_iris()
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(iris.data, iris.target)
  
  # Save for API
  joblib.dump(model, 'iris_model.pkl')
  ```
- **API Input:** 4 features (sepal length, sepal width, petal length, petal width)
- **API Output:** Iris species prediction
- **Why:** Simple, quick to set up, focus on API dev not training.

### Option 2: Your Own Trained Model 🔧
- **What:** Use a model from earlier weeks
- **Options:**
  - Churn predictor (Weeks 4-5)
  - House price predictor (Weeks 6-7)
  - Any model you trained previously
- **How to use:**
  ```python
  import pickle
  
  # Load your existing model
  with open('my_model.pkl', 'rb') as f:
      model = pickle.load(f)
  
  # Use in FastAPI
  @app.post("/predict")
  def predict(request: PredictionRequest):
      prediction = model.predict([[...]])
      return {"prediction": prediction}
  ```
- **Why:** Real practice with your own models, reinforces earlier learning.

### Option 3: Pre-trained Hugging Face Models 🤗 **FOR LLM/TEXT**
- **What:** Download ready-to-use models from Hugging Face
- **Examples:**
  ```python
  from transformers import pipeline
  
  # Sentiment analysis
  classifier = pipeline("sentiment-analysis")
  
  # Text summarization
  summarizer = pipeline("summarization")
  
  # Question answering
  qa = pipeline("question-answering")
  ```
- **How to use in API:**
  ```python
  @app.post("/sentiment")
  def analyze_sentiment(request: SentimentRequest):
      result = classifier(request.text)
      return {"sentiment": result}
  ```
- **Why:** Pre-built, production-ready, great for NLP.
- **Setup:** `pip install transformers torch`

### Option 4: Pre-trained Scikit-learn Models 🎓
- **What:** Simple classifiers/regressors from sklearn
- **Gallery of pre-built examples:**
  ```python
  from sklearn import datasets, svm
  
  # Example: SVM classifier
  X, y = datasets.load_digits(return_X_y=True)
  clf = svm.SVC(gamma=0.001)
  clf.fit(X, y)
  
  # Save and deploy
  import joblib
  joblib.dump(clf, 'digit_classifier.pkl')
  ```
- **Why:** Lightweight, portable, easy to serialize.

### Option 5: PyTorch/TensorFlow Models 🧠
- **What:** Deep learning models (if you want to serve deep networks)
- **Where to find:**
  - PyTorch Hub: `torch.hub.load()`
  - TensorFlow Hub: `https://www.tensorflow.org/hub`
- **Examples:**
  ```python
  import torch
  
  # Get pre-trained ResNet
  model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
  
  # Or TensorFlow
  import tensorflow_hub as hub
  model = hub.load("https://tfhub.dev/...")
  ```
- **Why:** Advanced, handles images/complex tasks.
- **Challenge:** Larger file sizes, more setup needed.

---

## 🚀 Quick Start Recommendation

**For fastest learning:** Use **Option 1** (Iris classifier)
- Train in 5 seconds
- Deploy in 10 minutes
- Focus on API design, not model training

**For depth:** Use **Option 2** (your own model)
- Reinforces earlier projects
- More realistic scenario
- Shows end-to-end workflow

---

## 📦 Model Serialization Reference

Save your model ONCE, load it many times:

```python
# SAVE (do once after training)
import joblib
joblib.dump(model, 'my_model.pkl')

# or pickle
import pickle
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# LOAD (in FastAPI)
import joblib
model = joblib.load('my_model.pkl')

# Use in prediction
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load('my_model.pkl')
```

---

## Concept 1: From Notebook to API

**Notebook:**
```python
# User has model locally
model = load_model()
prediction = model.predict([features])
print(prediction)
```

**API:**
```python
# User sends HTTP request
POST /predict
{"features": [2.5, 3.1, 1.2, ...]}

# Server responds
{"prediction": 0.87, "confidence": 0.94}
```

---

## Concept 2: FastAPI Basics

**What:** Modern Python framework for building APIs.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    features = [request.feature1, request.feature2, request.feature3]
    prediction = model.predict([features])[0]
    return {
        "prediction": float(prediction),
        "class": "cat" if prediction > 0.5 else "dog"
    }
```

**Run:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Test:**
```
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 2.5, "feature2": 3.1, "feature3": 1.2}'
```

---

## Concept 3: Request Validation

**What:** Automatic input checking.

```python
from pydantic import BaseModel, Field, validator

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=150, description="Age must be 0-150")
    income: float = Field(..., gt=0, description="Income must be positive")
    text: str = Field(..., min_length=3, max_length=1000)
    
    @validator('age')
    def age_must_be_reasonable(cls, v):
        if v > 120:
            raise ValueError('age must be < 120')
        return v

# FastAPI auto-validates:
# - POST {"age": -5} → Error: "must be >= 0"
# - POST {"age": 150} → Valid
# - POST {"age": "invalid"} → Error: "not a valid integer"
```

---

## Concept 4: Error Handling

**What:** Graceful error responses.

```python
from fastapi import HTTPException

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Load model (might fail)
        model = load_model()
        
        # Validate input
        if len(request.features) != 10:
            raise HTTPException(
                status_code=400,
                detail="Expected 10 features"
            )
        
        prediction = model.predict([request.features])[0]
        return {"prediction": float(prediction)}
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model not found on server"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )
```

---

## Concept 5: Model Serving Patterns

**What:** Different ways to structure API.

```python
# Pattern 1: Single model endpoint
@app.post("/predict/iris")
def predict_iris(features: IrisRequest):
    return iris_model.predict(features)

# Pattern 2: Multiple models
@app.post("/predict/{model_name}")
def predict(model_name: str, features: Features):
    if model_name == "iris":
        return iris_model.predict(features)
    elif model_name == "churn":
        return churn_model.predict(features)

# Pattern 3: Version endpoints
@app.post("/v1/predict")
def predict_v1(data: V1Request):
    return model_v1.predict(data)

@app.post("/v2/predict")
def predict_v2(data: V2Request):
    return model_v2.predict(data)
```

---

## Concept 6: Logging & Monitoring

**What:** Track API usage and errors.

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(request: PredictionRequest):
    logger.info(f"Prediction request: features={request.features}")
    
    try:
        result = model.predict([request.features])
        logger.info(f"Prediction succeeded: {result}")
        return {"prediction": float(result[0])}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500)

# Logs appear in console/files for debugging
```

---

## Concept 7: Containerization

**What:** Package API in Docker for consistent deployment.

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY app.py .
COPY model.pkl .

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t my-api:1.0 .

# Run
docker run -p 8000:8000 my-api:1.0

# Access
curl http://localhost:8000/health
```

---

## Concept 8: Scaling with Load Balancer

**What:** Multiple API instances behind load balancer.

```
       ┌─ API Instance 1 (port 8000)
Client → Load Balancer ─ API Instance 2 (port 8001)
       └─ API Instance 3 (port 8002)
```

```yaml
# docker-compose.yml
version: '3'
services:
  api:
    build: .
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

---

## Concept 9: Cloud Deployment

**What:** Deploying to cloud platforms.

```python
# Google Cloud Run
gcloud run deploy my-api \
  --source . \
  --platform managed \
  --region us-central1

# AWS Lambda
zappa init
zappa deploy production

# Heroku
git push heroku main
```

---

## Concept 10: API Documentation

**What:** Auto-generating docs for users.

```python
# FastAPI auto-generates Swagger docs!
# Visit: http://localhost:8000/docs

# Customize with docstrings
@app.post("/predict", responses={
    200: {"description": "Successful prediction"},
    400: {"description": "Invalid input"},
    500: {"description": "Server error"}
})
def predict(request: PredictionRequest):
    """
    Make a prediction.
    
    - **feature1**: First feature (0-1 range)
    - **feature2**: Second feature (0-1 range)
    
    Returns prediction probability.
    """
    ...
```

---

## Challenge Approach

### Challenge 1-3: Basic API
- Create FastAPI app with /health endpoint
- Add /predict endpoint for your model
- Test with curl/Postman

### Challenge 4-6: Validation & Error Handling
- Add input validation with Pydantic
- Handle model loading errors
- Test error cases

### Challenge 7-9: Logging & Containerization
- Add logging to track requests
- Create Dockerfile
- Test Docker build and run

### Challenge 10-12: Deployment
- Deploy to cloud (or local docker-compose)
- Document API with auto-docs
- Create deployment checklist

---

## Key Takeaways

✅ **FastAPI = fast, modern API framework** (async by default)

✅ **Pydantic validates inputs** (prevents bad data from reaching model)

✅ **Docker ensures reproducibility** (code + env packaged together)

✅ **Load balancing scales API** (handle many requests)

✅ **Cloud deployment makes accessible** (users hit API, not your laptop)
