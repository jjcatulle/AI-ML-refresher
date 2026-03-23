# Weekly Execution Checklist (Job-Ready 2026)

Use this as your weekly pass-fail scorecard.

## How To Use
- Weekly target time: 8 to 12 focused hours.
- Pass rule: finish the weekly deliverable plus 1 quality check.
- If you miss 2 weeks in a row, run a catch-up week before moving on.

## Phase 0: Beginner Foundation Ramp
- Week 0: ML 101 and workflow vocabulary. Pass if you can explain AI vs ML vs Data Science in plain language and sketch the ML lifecycle.
- Week 1: Python and Jupyter basics. Pass if you can run cells, define variables/functions, and debug a simple syntax error yourself.
- Week 2: Pandas deep practice. Pass if you can load CSVs, select rows/columns, group data, and explain what a DataFrame is.
- Week 3: NumPy and matrix intuition. Pass if you can explain array shape, transpose, and dot product using a small example.
- Week 4: Visualization plus first sklearn preview. Pass if you can create 4 chart types and explain what each chart tells you.

## Phase 1: Foundations
- Week 1: Load and profile one dataset. Pass if you produce a notebook with data types, missingness, and 5 key findings.
- Week 2: Visualization depth. Pass if you build a chart set and explain in words what each graph means.
- Week 3: Manual math-to-code day. Pass if Normal Equation and Gradient Descent both match sklearn within small error.
- Week 4: Build preprocessing pipeline. Pass if one reusable sklearn Pipeline runs end-to-end.
- Weeks 5-6: Churn model baseline plus tuned model. Pass if PR-AUC improves over baseline and confusion matrix is documented.
- Weeks 7-8: Regression/risk scoring project. Pass if you compare at least 3 models and explain residual diagnostics.
- Weeks 9-10: Review and harden. Pass if you publish one polished write-up in projects folder.

## Phase 2: Agentic LLM Systems
- Weeks 5-6: Prompt and tool fundamentals. Pass if chatbot supports 2 tools and error handling.
- Weeks 7-8: Hybrid retrieval. Pass if BM25 plus vector search is implemented with fused ranking.
- Weeks 9-10: RAG evaluation sprint. Pass if RAGAS metrics are logged and trend is tracked over 3 iterations.
- Weeks 11-12: Multi-agent orchestration. Pass if planner, researcher, writer, critic flow runs with retry or reflection.
- Weeks 13-14: Fine-tune vs RAG experiment. Pass if you present latency, quality, and cost tradeoff table.
- Weeks 15-16: Production rollout prep. Pass if shadow mode compares old/new outputs on real traffic samples.

## Phase 3: Deep Learning and Context Engineering
- Weeks 17-18: Attention and KV cache fundamentals. Pass if you can explain O(n^2) attention cost and cache memory tradeoff.
- Weeks 19-20: CNN baseline. Pass if model reaches target metric and training curves are saved.
- Weeks 21-22: Transfer learning. Pass if transfer model beats scratch model on both speed and accuracy.
- Weeks 23-24: Transformer-first NLP. Pass if transformer baseline beats LSTM baseline on the same task.
- Weeks 25-26: Multimodal context prototype. Pass if retrieval works for text plus at least one non-text modality.

## Phase 4: Production, Evals, Governance
- Weeks 24-26: Model API shipping. Pass if API has health, predict, and version endpoints plus request validation.
- Weeks 27-28: Infra hardening. Pass if Dockerized service runs locally with one-command startup.
- Weeks 29-31: Monitoring and evals. Pass if drift, latency, and quality dashboards are active.
- Weeks 32-34: Self-healing pipeline. Pass if retries, fallback model path, and alerting are tested.
- Weeks 35-37: Governance and guardrails. Pass if PII redaction, toxicity filter, and injection checks are in the request path.
- Weeks 38-39: Autonomous capstone. Pass if critic verification, guardrails, and auto-eval all run in one end-to-end workflow.

## Are You Overdoing It?
You are not overdoing it if these are true:
- You complete one strong deliverable per week, not three mediocre ones.
- You keep a stable cadence (8 to 12 hours) without burnout.
- You can explain your system design decisions clearly in plain language.

You are overdoing it if these are true:
- You keep adding tools but not finishing deployable artifacts.
- You skip evals and guardrails because of time pressure.
- You are consistently exhausted for more than 2 weeks.

## Do You Need More Data or More Lessons?
- More data: only when your current model is variance-limited and you have already fixed feature quality and eval quality.
- More lessons: when you cannot explain errors, tradeoffs, or architecture choices.
- Default in 2026: prioritize better evals and system reliability before chasing bigger datasets.

## Monthly Reality Check
- Month 1-2: Can I build and debug classic ML pipelines quickly?
- Month 3-4: Can I prove my RAG/agent got better with metrics?
- Month 5-6: Can I reason about transformer cost and latency tradeoffs?
- Month 7-9: Can I ship a safe, monitored, reliable autonomous system?