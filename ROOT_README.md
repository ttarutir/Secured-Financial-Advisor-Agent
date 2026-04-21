# Security-Aware Design Patterns for LLM-Based Financial Advisor Agents
### Measuring Robustness to Prompt Injection

**Takudzwa Tarutira & Lydia Abebe**  
Carnegie Mellon University · Secure and Responsible AI · Spring 2026

---

## Repository Structure

```
├── financial_advisor_baseline/   # Phase 1 — Baseline agent (no defenses)
├── financial_advisor_phase2/     # Phase 2 — Secured agent (Guard Node)
└── README.md                     # This file
```

---

## Phase 1 — Baseline Agent (`financial_advisor_baseline/`)

An unguarded LangGraph financial advisor agent built with Milvus and Gemini 2.5 Flash. The agent has no structural prompt injection defenses — it relies only on its system prompt to enforce policy compliance. This is the **vulnerability benchmark** used to measure Attack Success Rate (ASR) before any security mechanisms are applied.

**Key characteristics:**
- ReAct-style pipeline: retrieve → generate → finalize
- No input validation or chunk inspection
- Standard system prompt with policy instructions only
- FastAPI on port `8000`, Streamlit UI on port `8501`

---

## Phase 2 — Secured Agent (`financial_advisor_phase2/`)

Extends the baseline with a **three-layer Guard Node** inserted into the LangGraph execution pipeline. Both agents (baseline and secured) are included in this folder for side-by-side comparison against the same Milvus vector store.

**Guard Node layers:**
1. **Input Validation** — 22 regex patterns scan the user query before retrieval begins
2. **Chunk Inspection** — every retrieved document chunk is scanned before entering the LLM context
3. **Hardened System Prompt** — model explicitly instructed to ignore injected instructions and enforce Reg BI unconditionally

**Ports:**
| Service | Baseline | Secured |
|---|---|---|
| Agent API | `8000` | `8001` |
| Streamlit UI | `8501` | `8502` |

Also includes:
- `seed_data.py` — seeds Milvus with firm policies, client profiles, and market data
- `run_experiments.py` — automated benchmark runner (20 scenarios, ASR + utility metrics)
- `data/benchmark_scenarios.json` — 10 clean + 10 adversarial test scenarios

---

## Results

| Metric | Baseline | Secured |
|---|---|---|
| Attack Success Rate (ASR) | 30% | 10% |
| Block Rate | 0% | 90% |
| Utility on Clean Queries | 100% | 100% |
| False Positive Rate | 0% | 0% |

The Guard Node blocked 9 of 10 adversarial scenarios with zero false positives on legitimate client queries.

---

## Setup

Each folder contains its own `README.md` with full setup instructions. In both cases the stack runs via Docker Compose — no local Python environment needed beyond seeding the vector store.

```bash
# Quick start (Phase 2)
cd financial_advisor_phase2
cp .env.example .env        # add your GOOGLE_API_KEY
docker compose up --build
python seed_data.py
# Baseline UI → http://localhost:8501
# Secured UI  → http://localhost:8502
```
