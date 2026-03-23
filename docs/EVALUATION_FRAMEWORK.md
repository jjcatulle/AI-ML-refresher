# Evaluation Framework
**How to prove your ML/AI system is actually getting better**

---

## Why This Exists

A lot of AI/ML learning plans tell you to build models, prompts, or agents.
Very few teach you how to measure quality in a repeatable way.

That is a problem.

If you cannot evaluate a system, you cannot:
- compare two model versions,
- know whether a change improved anything,
- catch regressions before deployment,
- explain quality to a hiring manager or teammate,
- ship safely in production.

This workspace treats evaluation as a core engineering skill, not an afterthought.

---

## The 6-Layer Evaluation Framework

Use these layers in every serious project.

### 1. Task Definition
Before you train or prompt anything, write down:
- the exact task,
- the desired output format,
- what a good answer or prediction looks like,
- what failure looks like,
- what metric matters most.

Examples:
- Churn model: rank likely churners so retention team can intervene.
- House price model: minimize pricing error on unseen homes.
- RAG bot: answer from retrieved sources without hallucinating.
- Agent system: complete a multi-step task while staying within tool and safety constraints.

If the task definition is vague, every metric after that becomes unreliable.

### 2. Offline Ground-Truth Evaluation
This is the first hard check.

You need a held-out dataset or labeled test set.

For classical ML:
- classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
- regression: RMSE, MAE, R-squared
- ranking/recommendation: recall@k, precision@k, MAP, NDCG

For RAG/LLM systems:
- exact-match or reference-answer checks where possible
- retrieval recall@k
- answer correctness against known ground truth
- citation/source correctness

Rule:
- never evaluate on the same data you used to fit or tune the system.

### 3. Slice-Based Error Analysis
Overall averages hide failure modes.

After measuring the full dataset, break results down by slices such as:
- class label,
- customer segment,
- short vs long documents,
- easy vs hard examples,
- low-resource categories,
- adversarial or noisy inputs.

If you only track one aggregate score, you will miss the real problem.

### 4. Baseline and Regression Testing
Every project needs a baseline.

Compare against:
- a simple heuristic,
- a previous model,
- a smaller prompt,
- a plain keyword search system,
- a simpler non-agent workflow.

Then create regression tests:
- fixed examples that must keep passing,
- edge cases that previously failed,
- safety and refusal cases,
- adversarial prompts,
- noisy real-world inputs.

If a new version beats the average metric but breaks known critical cases, it is not ready.

### 5. System-Level Evaluation
This layer matters most for LLM, RAG, and agent systems.

Measure more than model output quality:
- retrieval quality,
- tool selection accuracy,
- latency,
- cost per request,
- citation quality,
- hallucination rate,
- refusal quality,
- policy compliance.

Useful tools in this workspace:
- `ragas` for RAG quality metrics
- `mlflow` or `wandb` for experiment tracking
- `evidently` for drift and monitoring
- LLM-as-a-Judge for structured qualitative comparisons

### 6. Production Evaluation
Offline eval is necessary, but not sufficient.

Once deployed, quality can drift.

Track:
- live latency,
- failure rate,
- cost,
- user feedback,
- input drift,
- output drift,
- online quality proxies,
- shadow mode comparisons between old and new systems.

Production rules:
- deploy new versions behind a gate,
- compare old/new on real traffic in shadow mode,
- monitor regressions before full rollout,
- keep a rollback path.

---

## What to Evaluate by System Type

### Classical ML
Minimum bar:
- clear train/validation/test split,
- baseline model,
- core metric,
- confusion matrix or residual analysis,
- feature importance or error analysis,
- cross-validation where appropriate.

### RAG Systems
Minimum bar:
- retrieval recall@k,
- answer correctness,
- citation correctness,
- hallucination checks,
- failure cases logged in an eval set.

### Agent Systems
Minimum bar:
- task success rate,
- tool-use correctness,
- step completion rate,
- safety failures,
- latency/cost budget,
- benchmark task suite.

### Production Systems
Minimum bar:
- experiment tracking,
- release criteria,
- shadow mode,
- monitoring dashboard,
- drift checks,
- rollback procedure.

---

## Evaluation Deliverables by Phase

### Phase 0
- Write down the task and success metric before each notebook.
- Explain why one metric is better than another for the task.

### Phase 1
- Build proper offline evaluation habits.
- Use train/test split, cross-validation, confusion matrices, residual plots.
- Add error analysis and threshold thinking.

### Phase 2
- Create an `evals/` folder for gold question-answer sets.
- Score retrieval and answer quality.
- Add RAGAS and LLM-as-a-Judge experiments.
- Track regressions across prompt and retrieval changes.

### Phase 3
- Evaluate model quality by class, dataset split, and failure slices.
- Track training vs validation behavior.
- Compare from-scratch vs transfer learning fairly.

### Phase 4
- Add production monitoring, shadow mode, and release gates.
- Track quality, cost, latency, and safety together.

---

## Default Evaluation Workflow for Any Project

1. Define the task and write the primary metric.
2. Create a baseline.
3. Build a held-out eval set.
4. Measure offline quality.
5. Break results into slices.
6. Save failure cases into regression tests.
7. Compare variants in a structured table.
8. Only then deploy.
9. Monitor online behavior and keep rollback ready.

---

## Example Scorecards

### Churn Predictor Scorecard
- Primary metric: recall on churned customers
- Secondary metrics: precision, PR-AUC, calibration, confusion matrix
- Slices: contract type, tenure band, monthly charge band
- Baseline: logistic regression vs random forest
- Regression set: known borderline customers and minority churn cases

### RAG Bot Scorecard
- Primary metric: answer correctness
- Secondary metrics: faithfulness, retrieval recall@k, citation correctness, latency, cost
- Slices: short docs, long docs, ambiguous questions, adversarial prompts
- Baseline: keyword-only retrieval vs hybrid retrieval vs hybrid + reranker
- Regression set: hallucination traps, missing-context prompts, injection attempts

### Agent Scorecard
- Primary metric: task completion rate
- Secondary metrics: tool correctness, retries, latency, safety failures, cost per task
- Slices: single-tool, multi-tool, long-horizon, ambiguous instructions
- Baseline: fixed workflow vs autonomous agent
- Regression set: broken tools, malformed inputs, safety-boundary tasks

---

## What Good Looks Like in 2026

By the end of this roadmap, you should be able to say:

- I do not ship models without a baseline.
- I do not trust one aggregate metric.
- I keep an eval set and a regression set.
- I evaluate retrieval, generation, and agent behavior separately.
- I compare versions before rollout.
- I use shadow mode and monitoring in production.

That is the difference between a demo builder and a production AI engineer.
