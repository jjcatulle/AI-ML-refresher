# Setup Guide: Getting Started with ML-AI Learning

Complete walkthrough to setup the project and start working with Jupyter notebooks.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Installation Steps](#installation-steps)
4. [Running Notebooks](#running-notebooks)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:
- **Python 3.8+** installed on your system
  - Check: `python3 --version`
- **Git** (optional, for cloning)
  - Check: `git --version`
- **Terminal/Command Line** access
- **Text Editor or IDE** (VS Code, PyCharm, or similar)

### Python Installation

**macOS:**
```bash
# Using Homebrew
brew install python3

# Verify
python3 --version
```

**Windows:**
- Download from https://www.python.org/downloads/
- Run installer and select ✅ "Add Python to PATH"
- Verify: `python --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

---

## Popular Python ML/AI Packages To Install

These are the most common packages used by ML/AI engineers in 2026.

### Core Foundation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Strong Classical ML
```bash
pip install xgboost lightgbm catboost
```

### Deep Learning
```bash
pip install torch torchvision torchaudio
pip install tensorflow
```

### NLP and LLM
```bash
pip install transformers tokenizers datasets sentence-transformers peft
```

### Retrieval and Vector Search
```bash
pip install faiss-cpu rank-bm25
pip install qdrant-client weaviate-client pinecone-client
```

### Agentic and App Framework
```bash
pip install langchain llama-index fastapi uvicorn pydantic
```

### Evals, Monitoring, and Tracking
```bash
pip install ragas mlflow wandb evidently
```

### Data Quality and Pipelines
```bash
pip install great-expectations prefect dbt-core
```

### Optional Performance and Scale
```bash
pip install polars duckdb ray pyspark
```

### Beginner Advice
- Install in layers, not all at once.
- Start with foundation packages first.
- Add advanced libraries only when a project needs them.
- Keep your environment in `requirements.txt` for reproducibility.

---

## Project Structure

```
ML-AI-learning/
├── README.md                    # Project overview
├── docs/                        # Documentation
│   ├── SETUP_GUIDE.md          # This file
│   ├── CHALLENGE_TRACKER.md    # Progress tracker
│   └── QUICK_REFERENCE.md      # Code cheat sheet
├── guides/                      # Learning materials (read these!)
│   ├── common/HOW_TO_SOLVE_CHALLENGES.md
│   ├── phase0/                 # Foundation guides
│   ├── phase1/                 # Phase 1 guides
│   ├── phase2/                 # Phase 2 guides
│   ├── phase3/                 # Phase 3 guides
│   └── phase4/                 # Phase 4 guides
├── phases/                   # Your work folder
│   ├── phase0/starters/        # Foundation starter notebooks
│   ├── phase1/starters/        # Phase 1 starter notebooks + your solutions
│   ├── phase2/starters/        # Phase 2 starter notebooks
│   ├── phase3/starters/        # Phase 3 starter notebooks
│   └── phase4/starters/        # Phase 4 starter notebooks
├── data/                        # Store datasets
├── models/                      # Save trained models
├── .venv/                       # Virtual environment (created later)
└── requirements.txt             # Dependencies
```

---

## Installation Steps

### Step 1: Navigate to Project Directory

**macOS/Linux:**
```bash
cd ~/Desktop/ML-AI-learning
```

**Windows:**
```bash
cd C:\Users\YourUsername\Desktop\ML-AI-learning
```

### Step 2: Create Virtual Environment

A virtual environment isolates project dependencies from your system Python.

**macOS/Linux:**
```bash
python3 -m venv .venv
```

**Windows:**
```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

Your prompt should now show `(.venv)` at the start.

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- Jupyter & JupyterLab
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- PyTorch / TensorFlow
- FastAPI, requests
- And more...

**Installation time:** 5-15 minutes (depends on internet speed)

### Step 6: Verify Installation

```bash
python -c "import jupyter, pandas, numpy, sklearn; print('✅ All packages installed!')"
```

If successful, you'll see: `✅ All packages installed!`

---

## Running Notebooks

### Option 1: Jupyter Lab (Recommended)

**Start Jupyter Lab:**
```bash
jupyter lab
```

This opens http://localhost:8888 in your browser.

**Navigate to notebooks:**
1. Click on `phases/phase1/starters/` folder in left sidebar
2. Double-click any `STARTER_*.ipynb` file
3. Notebook opens in the editor

### Option 2: Jupyter Notebook

**Start Jupyter Notebook:**
```bash
jupyter notebook
```

This opens http://localhost:8888 in your browser.

**Navigate and open:**
1. Navigate to `phases/phase1/starters/` folder
2. Click on notebook name to open

### Option 3: VS Code

1. Install "Jupyter" extension in VS Code
2. Open notebook file from a phase starter folder, for example: `File → Open → phases/phase1/starters/STARTER_Week1_DataExploration.ipynb`
3. Notebooks open directly in editor

---

## Working with Notebooks

### Reading the Guide

Before starting a challenge notebook:

1. Read the corresponding guide:
   ```
   Week 1: guides/phase1/GUIDE_Week1_DataExploration.md
   Weeks 4-5: guides/phase1/GUIDE_Weeks4-5_ChurnPredictor.md
   (etc.)
   ```

2. Guide teaches concepts and shows examples

### Solving Challenges

In each notebook:

1. **Read the markdown cells** - These explain each challenge
2. **Write code in code cells** - Replace "YOUR CODE HERE"
3. **Run cells** - Press `Shift + Enter` to execute
4. **Verify output** - Check results below cell

### Example Challenge

```
## Challenge 1: Load Data

**Goal:** Load a dataset and explore it

**Steps:**
1. Import pandas
2. Load CSV file
3. Display shape and first 5 rows
```

**Your solution:**
```python
# Challenge 1: Load Data

import pandas as pd

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Display shape
print(df.shape)

# Display first rows
print(df.head())
```

### Save Your Work

**Save your solution notebook:**
```
File → Save As → phases/phase1-4/Week1_MySolution.ipynb
```

This preserves your work and keeps starters clean.

---

## Running Your First Notebook

### Quick Start (5 minutes)

1. **Activate environment:**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Start Jupyter:**
   ```bash
   jupyter lab
   ```

3. **Open notebook:**
   - Click `phases/phase1/starters/STARTER_Week1_DataExploration.ipynb`

4. **Read first challenge**

5. **Write code** in code cells

6. **Press `Shift + Enter`** to run

---

## Deactivating Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

Your prompt will return to normal (no `(.venv)` prefix).

---

## Troubleshooting

### Issue: "Python not found"

**Solution:** Ensure Python is installed and in PATH
```bash
python3 --version
```

If not found, install Python (see Prerequisites section).

### Issue: "No module named 'jupyter'"

**Solution:** Install Jupyter
```bash
pip install jupyter jupyterlab
```

### Issue: "Permission denied" or "Access denied"

**Solution:** Ensure you're in correct directory and venv is activated
```bash
pwd                    # Check current directory
source .venv/bin/activate  # Reactivate venv
```

### Issue: Notebook won't open in VS Code

**Solution:** Install Jupyter extension
1. Open VS Code
2. Extensions icon (left sidebar)
3. Search "Jupyter"
4. Install (Microsoft)
5. Reload VS Code

### Issue: Slow Jupyter startup

**Solution:** Use Jupyter Lab instead or clear cache
```bash
jupyter lab --no-browser
# Then copy URL and paste in browser
```

### Issue: "Port 8888 already in use"

**Solution:** Use different port
```bash
jupyter lab --port 8889
```

Then open http://localhost:8889

### Issue: Dependencies fail to install

**Solution:** Try installing one at a time
```bash
pip install jupyter
pip install pandas
pip install numpy
# (etc.)
```

Or check your internet connection and retry:
```bash
pip install -r requirements.txt --verbose
```

---

## Next Steps

1. ✅ Complete setup (this guide)
2. 📖 Read Week 1 guide: `guides/phase1/GUIDE_Week1_DataExploration.md`
3. 📓 Open notebook: `phases/phase1/starters/STARTER_Week1_DataExploration.ipynb`
4. 💻 Solve challenges 1-12
5. 📊 Progress tracker: `docs/CHALLENGE_TRACKER.md`

---

## Getting Help

If stuck:

1. **Check the guide** - `guides/phase*/GUIDE_*.md` has concepts and examples
2. **Read error messages** - They tell you what's wrong
3. **Google the error** - Most Python errors are documented online
4. **Try a simpler version** - Break down complex problems
5. **Take a break** - Fresh perspective helps!

---

## Environment Maintenance

### Update packages (optional)

```bash
pip install --upgrade -r requirements.txt
```

### Check what's installed

```bash
pip list
```

### Remove virtual environment (to restart)

```bash
rm -rf .venv  # macOS/Linux
rmdir /s .venv  # Windows
```

Then create a fresh one (Step 2 above).

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Activate environment | `source .venv/bin/activate` |
| Deactivate | `deactivate` |
| Start Jupyter Lab | `jupyter lab` |
| Start Jupyter Notebook | `jupyter notebook` |
| Install packages | `pip install package_name` |
| List installed | `pip list` |
| Check Python version | `python3 --version` |
| Check pip version | `pip --version` |

---

## Additional Resources

- **Python Docs:** https://docs.python.org/3/
- **Pandas Docs:** https://pandas.pydata.org/docs/
- **Jupyter Docs:** https://jupyter.org/
- **NumPy Docs:** https://numpy.org/doc/
- **Scikit-learn Docs:** https://scikit-learn.org/stable/documentation

---

**Congratulations! You're ready to start learning. 🚀**

Begin with Week 1: Read the guide, then solve the notebook challenges!
