# 🚀 ML/AI Learning Workspace

Your personalized zero-to-job-ready journey to become a production-ready AI Systems Architect.

---

## ⚡ Quick Start

**New to this workspace?** Start here:

1. **[📖 SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Complete setup instructions
   - Install Python & create virtual environment
   - Install dependencies
   - Install popular Python ML/AI engineer stack
   - Run Jupyter notebooks
   - Troubleshooting tips

2. **[🎯 HOW_TO_SOLVE_CHALLENGES.md](guides/common/HOW_TO_SOLVE_CHALLENGES.md)** - Learn the methodology
   
3. **[📊 CHALLENGE_TRACKER.md](docs/CHALLENGE_TRACKER.md)** - Track your progress

4. **[🧰 QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Includes the popular Python ML/AI stack used in industry

5. **[📏 EVALUATION_FRAMEWORK.md](docs/EVALUATION_FRAMEWORK.md)** - The scoring system for ML, RAG, and agent quality

6. **[🧼 DATA_PREP_CHEATSHEET.md](docs/DATA_PREP_CHEATSHEET.md)** - Repeatable pre-training data preparation checklist

7. **[🔁 FULL_ML_FLOW_CHECKLIST.md](docs/FULL_ML_FLOW_CHECKLIST.md)** - End-to-end project flow habit from problem to handoff

8. **[🗂️ ONE_PAGE_ML_FLOW_CARD.md](docs/ONE_PAGE_ML_FLOW_CARD.md)** - Detailed execution card with examples, checks, and references

9. **[🎛️ HYPERPARAMETER_TUNING_GUIDE.md](docs/HYPERPARAMETER_TUNING_GUIDE.md)** - How tuning works, when to do it, and how to avoid tuning mistakes

Then start with Week 1: Read `guides/phase1/GUIDE_Week1_DataExploration.md` → Open `phases/phase1/starters/STARTER_Week1_DataExploration.ipynb`

If you are brand new to Python or ML, start with the deeper foundation ramp in `docs/AI-ML-LEARNING-GUIDE.md` before rushing into projects.

---

## 📁 Folder Structure

```
ML-AI-learning/
├── README.md                                  # This file - START HERE
│
├── 📚 guides/                                 # Learning guides (read BEFORE challenges)
│   ├── common/
│   │   └── HOW_TO_SOLVE_CHALLENGES.md        # 🎯 METHODOLOGY - READ THIS!
│   ├── phase0/                               # Phase 0 guide files
│   ├── phase1/                               # Phase 1 guide files
│   ├── phase2/                               # Phase 2 guide files
│   ├── phase3/                               # Phase 3 guide files
│   └── phase4/                               # Phase 4 guide files
│
├── 📖 docs/                                   # Documentation & references
│   ├── AI-ML-LEARNING-GUIDE.md                # 16-page curriculum overview
│   ├── EVALUATION_FRAMEWORK.md                # How to measure ML/RAG/agent quality
│   ├── DATA_PREP_CHEATSHEET.md                # Data preparation checklist before modeling
│   ├── FULL_ML_FLOW_CHECKLIST.md              # Full ML project flow checklist
│   ├── ONE_PAGE_ML_FLOW_CARD.md               # Detailed step-by-step ML execution card
│   ├── HYPERPARAMETER_TUNING_GUIDE.md         # Explanation-first tuning guide
│   ├── QUICK_REFERENCE.md                     # Code syntax cheat sheet
│   ├── CHALLENGE_TRACKER.md                   # Your progress checklist
│   └── .agent.md                              # Your AI learning agent
│
├── 📁 YOUR WORK (Add here!)
│   ├── phases/
│   │   ├── phase0/
│   │   │   └── starters/             # Foundation starter notebooks
│   │   ├── phase1/
│   │   │   ├── starters/             # Phase 1 starter notebooks
│   │   │   └── MY_*.ipynb            # Your solved notebooks
│   │   ├── phase2/
│   │   │   └── starters/             # Phase 2 starter notebooks
│   │   ├── phase3/
│   │   │   └── starters/             # Phase 3 starter notebooks
│   │   └── phase4/
│   │       └── starters/             # Phase 4 starter notebooks
│   │
│   ├── data/                         # Datasets you download (keep clean!)
│   │   ├── kaggle_churn_dataset.csv
│   │   ├── iris.csv
│   │   └── [your data files]
│   │
│   ├── models/                       # Trained models you save
│   │   ├── churn_model.pkl
│   │   ├── image_classifier.pt
│   │   └── [your model files]
│   │
│   └── projects/                     # Completed projects for portfolio
│       ├── churn_predictor/
│       ├── rag_chatbot/
│       └── image_classifier/
│
├── 🧪 evals/                         # Gold datasets + eval outputs (RAGAS/Judge)
├── 🛠️ scripts/                       # Utility scripts (ETL, scraping, eval automation)
├── 🏗️ infra/                         # Docker Compose, Vector DB, deployment configs
│
└── 🔧 CONFIGURATION
    ├── .venv/                        # Your Python environment
    └── requirements.txt              # (Optional - list of packages)
```

## 📋 How This Works

### Choose Your Path First

1. **Absolute Beginner Path**
   - Start with the foundation ramp in `docs/AI-ML-LEARNING-GUIDE.md`
   - Work slowly through beginner sections in guides and starters
   - Treat the first month as skills + concepts + workflow training

2. **Fast Track Path**
   - If you already code comfortably, compress the beginner ramp
   - Move into Week 1 and Phase 1 projects faster

### For Each Challenge Set:

0. **Read EVALUATION_FRAMEWORK.md** (in docs/)
   - Know what success metric you are optimizing
   - Keep a baseline and failure cases
   - Treat evals as part of the build, not cleanup at the end

0.5 **Read DATA_PREP_CHEATSHEET.md and FULL_ML_FLOW_CHECKLIST.md** (in docs/)
   - Prepare data before visualization and training
   - Follow a consistent full-cycle workflow on every project

0.6 **Use ONE_PAGE_ML_FLOW_CARD.md during execution** (in docs/)
   - Follow the expanded step-by-step checks
   - Use the embedded examples and references while building
   - Build repeatable habits instead of one-off fixes

1. **Read HOW_TO_SOLVE_CHALLENGES.md** (in guides/common/)
   - 7-step methodology for solving ANY challenge
   - Common mistakes to avoid
   - Tips for success

2. **Read the Relevant Guide** (e.g., guides/phase1/GUIDE_Week1_DataExploration.md)
   - Explains every concept you'll need
   - Shows example approaches (not complete code)
   - Answers "Why?" for each concept

3. **Open the Starter Notebook** (e.g., phases/phase1/starters/STARTER_Week1_DataExploration.ipynb)
   - Each challenge has a description
   - Hints provided (but not solutions)
   - "YOUR CODE HERE" placeholders
   - You write every line!

4. **Save Your Solutions** to phases/phase1/
   - Rename when complete: `My_Week1_DataExploration.ipynb`
   - This becomes your portfolio!

5. **Check docs/QUICK_REFERENCE.md** if stuck on syntax
   - But don't copy-paste - type it out!

6. **Move to Next Challenge**
   - Mark it done in docs/CHALLENGE_TRACKER.md
   - Review what you learned
   - Then tackle next one

---

## 🎯 Current Progress

### ✅ Completed
- [x] Dev environment setup (Python 3.9.6 + 20+ libraries)
- [x] AI/ML Learning Curriculum (16-page guide)
- [x] Custom Learning Agent (.agent.md)
- [x] **Challenge methodology guide** (`guides/common/HOW_TO_SOLVE_CHALLENGES.md`)
- [x] Phase 1, Week 1: Data Exploration Starter (10 challenges)
- [x] Phase 1, Weeks 4-5: Churn Predictor Starter (12 challenges)
- [x] **Learning guides** (`guides/phase1/GUIDE_Week1_DataExploration.md` + `guides/phase1/GUIDE_Weeks4-5_ChurnPredictor.md`)
- [x] Folder structure organized

### 🔄 This Week
- [ ] **READ**: guides/common/HOW_TO_SOLVE_CHALLENGES.md (methodology)
- [ ] **READ**: guides/phase1/GUIDE_Week1_DataExploration.md (concepts)
- [ ] **CODE**: Solve all 10 Week 1 challenges
- [ ] **SAVE**: Your solutions to phases/phase1/

### 📅 Next (Weeks 3+)
- [ ] Create own notebook with Kaggle dataset
- [ ] **READ**: guides/phase1/GUIDE_Weeks4-5_ChurnPredictor.md
- [ ] **CODE**: Solve all 12 Churn Predictor challenges
- [ ] Start Phase 2 (LLMs & RAG)

---

## 🚀 Quick Start

### Step 1: Understand the Process
Read: **guides/common/HOW_TO_SOLVE_CHALLENGES.md** (15 min)
- Learn the 7-step methodology
- Understand what each guide contains
- See an example challenge walkthrough

### Step 2: Learn the Concepts
Read: **guides/phase1/GUIDE_Week1_DataExploration.md** (30 min)
- All 10 concepts explained
- Why each matters
- Approach for each challenge

### Step 3: Solve the Challenges
Open: **phases/phase1/starters/STARTER_Week1_DataExploration.ipynb**
- Work through 10 challenges
- Use the methodology from Step 1
- Reference the guide from Step 2
- Check docs/QUICK_REFERENCE.md for syntax if stuck

### Step 4: Save Your Work
Create notebook: `phases/phase1/My_Solution_Week1.ipynb`
- Copy your finished notebook here
- Rename it clearly
- This becomes your portfolio!

### Step 5: Move to Next Challenge Set
When Week 1 is done, start Weeks 4-5:
- Read: guides/phase1/GUIDE_Weeks4-5_ChurnPredictor.md
- Open: phases/phase1/starters/STARTER_Weeks4-5_ChurnPredictor.ipynb
- Solve 12 challenges (complete ML project!)
- Save to phases/phase1/My_Solution_ChurnPredictor.ipynb

---

## 📚 The Learning Path

### ⏱️ Time Estimate

| Week | Challenge | # Challenges | Time | Output |
|------|-----------|---|---|---|
| **0** | ML 101 + Workflow | - | 3-5 hours | Concept notes + ML vocabulary |
| **1** | Python + Jupyter + Data Exploration | 10 | 6-8 hours | Your analysis on any dataset |
| **2** | Pandas/Charts Deep Practice | - | 6-8 hours | Data cleaning + visual explanations |
| **3** | NumPy + Math-to-Code | - | 6-8 hours | Manual regression lab |
| **4** | Sklearn Foundations | - | 6-8 hours | First pipeline + tiny model |
| **5-6** | Churn Predictor | 12 | 8-12 hours | Your first ML model |
| **7-8** | House Price | 12 | 8-12 hours | Regression model |
| **9-10** | Fraud Detection | 12 | 8-12 hours | Imbalanced classification model |

---

### ✅ Before You Start

- [ ] Read guides/common/HOW_TO_SOLVE_CHALLENGES.md
- [ ] Have QUICK_REFERENCE.md open
- [ ] Read the relevant guides/phase*/GUIDE_*.md file
- [ ] Have 1-2 hours blocked (no interruptions)
- [ ] Have notebook and pen nearby (for notes!)
- [ ] Ready to struggle a bit (that's learning!)

---

## 📖 Learning Resources

### Phase 1: ML Fundamentals
- 🎓 Kaggle Learn (5 hrs, free)
- 📚 Scikit-learn official docs
- 🎥 StatQuest with Josh Starmer (YouTube)

### Phase 2: LLMs (Your Fast Track!)
- 🎓 DeepLearning.AI Short Courses (5 hrs, free)
- 📚 LangChain Documentation
- 💻 Your own RAG chatbot project

### Phase 3: Deep Learning
- 🎓 Fast.ai Part 1 (40 hrs, free)
- 📚 PyTorch tutorials
- 🧠 Build CNN, LSTM, Transformers, and multimodal retrieval prototypes

### Phase 4: MLOps
- 🎓 Full Stack Deep Learning (16 hrs, free)
- 🐳 Docker + Cloud platform docs
- 🚀 Deploy real models to production

---

## 🧠 Learning Guides Explained

### **guides/common/HOW_TO_SOLVE_CHALLENGES.md** - Your Methodology
The meta-guide teaching you HOW to solve ANY challenge.

**Includes**:
- 7-step challenge-solving framework
- Difficulty levels (green/yellow/red)
- Common mistakes to avoid
- Pro tips for success
- Complete worked example

**Read when**: FIRST - before any challenges

---

### **guides/phase1/GUIDE_Week1_DataExploration.md** - Concept Teaching
Teaches you the 10 concepts needed for Week 1 challenges.

**Includes**:
- Concept 1-10 (import → insights)
- What each concept is
- How it works
- Why it matters
- Approach guidance (not solutions!)
- Challenge sequence

**Read when**: Before solving Week 1 challenges

---

### **guides/phase1/GUIDE_Weeks4-5_ChurnPredictor.md** - Project Teaching
Teaches you how to build a complete ML project.

**Includes**:
- What's the churn problem? (business context)
- The full ML pipeline (10 steps)
- Each step explained deeply
- Metrics explained with examples
- Common mistakes to avoid
- Challenge sequence

**Read when**: After Week 1, before Churn Predictor

---

### **QUICK_REFERENCE.md** - Code Syntax
Cheat sheet for common code patterns.

**Includes**:
- Pandas methods (load, explore, group)
- NumPy operations
- Matplotlib plotting
- Scikit-learn workflows
- FastAPI syntax
- Docker commands

**Use when**: Stuck on syntax or can't remember a function name

---

### **CHALLENGE_TRACKER.md** - Your Progress
Checklist of all challenges and milestones.

**Includes**:
- All 4 phases broken down
- All challenges listed
- Monthly milestones
- Success metrics
- Completion checkboxes

**Update when**: You finish each challenge

---

## 🎯 The "Read First" Order

1. **This README** (you're reading it!) ← You are here
2. **guides/common/HOW_TO_SOLVE_CHALLENGES.md** (15 min)
3. **guides/phase1/GUIDE_Week1_DataExploration.md** (30 min)
4. **phases/phase1/starters/STARTER_Week1_DataExploration.ipynb** (solve challenges)
5. **QUICK_REFERENCE.md** (reference as needed)
6. Repeat for next challenge set

---

## 💡 Why This Structure Works

### ✅ You Read to Understand
- Guides explain concepts deeply
- Methodology teaches HOW to solve
- No copy-paste solutions to be lazy with

### ✅ You Code to Learn
- Challenges require your own solutions
- You struggle (essential part of learning!)
- You understand what you wrote

### ✅ You Reference While Solving
- Quick reference for syntax
- Guides for concepts
- Methodology for approach

### ✅ You Track Your Progress
- Challenge tracker shows your achievements
- Visible milestones celebrate wins
- Clear path forward

---

## 🆘 If You're Stuck

### Stuck Understanding a Concept?
→ Read the relevant guides/phase*/GUIDE_*.md file again

### Stuck on Syntax?
→ Check QUICK_REFERENCE.md for examples

### Stuck on Approach?
→ Review guides/common/HOW_TO_SOLVE_CHALLENGES.md methodology

### Stuck on an Error?
→ 1. Read the error carefully
→ 2. Google the error + library name  
→ 3. Try simpler code
→ 4. Ask in communities

### Stuck Overall?
→ You're not alone! Struggle is where learning happens.
→ Take a break, comeback fresh
→ Re-read the guide
→ Try the example from the guide
→ Then your challenge

---

## 📊 Success Looks Like

### Week 1
- ✅ All 10 challenges completed
- ✅ Your own clean code
- ✅ You can explain each solution
- ✅ Saved to phases/phase1/

### Week 4-5
- ✅ All 12 challenges completed
- ✅ Model has >80% accuracy
- ✅ You identified top features
- ✅ Prediction function works
- ✅ Can explain to non-technical person

### By Month 3
- ✅ 3+ complete projects (portfolio)
- ✅ GitHub repos with clean code
- ✅ Can talk through each project
- ✅ Ready to show employers

---

## 🚀 Let's Get Started!

### This Week's Quest
- **Read**: guides/common/HOW_TO_SOLVE_CHALLENGES.md (15 min)
- **Read**: guides/phase1/GUIDE_Week1_DataExploration.md (30 min)  
- **Code**: 10 challenges in phases/phase1/starters/STARTER_Week1_DataExploration.ipynb (4-6 hrs)
- **Save**: Your solutions to phases/phase1/

### Your First Challenge Waiting
Open: `phases/phase1/starters/STARTER_Week1_DataExploration.ipynb`
Challenge 1: Import Libraries

**You've got this!** 💪

---

*Last Updated: March 18, 2026*
*Ready? Let's build your AI/ML career!* 🚀
