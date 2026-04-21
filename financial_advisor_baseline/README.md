# Financial Advisor Agent — Baseline (No Guard Node)

LangGraph + Milvus + Streamlit financial advisor agent.  
This is **Phase 1** of the security research project: the vulnerable baseline used to measure Attack Success Rate (ASR) before the Guard Node is introduced.

---

## Project Structure

```
financial_advisor/
├── src/
│   ├── agent/
│   │   ├── state.py        # AgentState schema
│   │   ├── nodes.py        # LangGraph nodes (retrieve → generate → finalize)
│   │   ├── graph.py        # LangGraph pipeline definition
│   │   └── server.py       # FastAPI server
│   ├── rag/
│   │   ├── embeddings.py   # Sentence Transformers embeddings
│   │   ├── indexer.py      # Text chunking
│   │   └── retriever.py    # Milvus vector store (financial_docs collection)
│   └── app/
│       └── streamlit_app.py
├── data/
│   └── synthetic_documents.json   # Firm policies + 3 malicious injection docs
├── seed_data.py            # Seeds Milvus with synthetic documents
├── docker-compose.yml      # Milvus (etcd + minio + milvus) + agent + streamlit
├── Dockerfile.agent
├── Dockerfile.streamlit
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Build and start all services

```bash
docker compose up --build
```

This starts:
| Service    | URL                    | Description                        |
|------------|------------------------|------------------------------------|
| Milvus     | localhost:19530        | Vector database                    |
| Agent API  | http://localhost:8000  | FastAPI + LangGraph agent          |
| Streamlit  | http://localhost:8501  | Web UI                             |

Milvus takes ~30 seconds to become healthy on first boot. The agent will retry automatically.

### 3. Seed the vector store

**Clean baseline (firm policies only):**
```bash
python seed_data.py
```

**With malicious injection documents (ASR testing):**
```bash
python seed_data.py --with-injections
```

**Reset and re-seed:**
```bash
python seed_data.py --reset --with-injections
```

### 4. Open the UI

Go to **http://localhost:8501**

---

## How to Run Experiments

### Clean baseline test
Seed with `python seed_data.py` (no injections).  
Ask normal client queries and verify the agent gives policy-compliant recommendations.

### ASR measurement (indirect injection)
Seed with `python seed_data.py --with-injections`.  
The 3 malicious documents are now in the vector store and will be retrieved alongside real policies.  
Ask the same clean client queries — observe whether the injected instructions affect the output.

### Direct injection test
Use the example query in the Streamlit UI labeled **"⚠️ Direct injection"**, or craft your own adversarial user queries.

---

## The 3 Injection Documents

| Document Title | Injection Type | Attack Goal |
|---|---|---|
| Market Outlook Q2 2024 — Special Advisory Notice | `direct_policy_override` | Force unsuitable product recommendation, bypass Reg BI |
| Client Risk Tolerance Assessment Results | `data_exfiltration` | Cause agent to leak client NPI from session context |
| Regulatory Update Memo — FINRA Notice 24-08 | `jailbreak_via_roleplay` | Override system identity, bypass all policy constraints |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Milvus collection stats |
| POST | `/query` | Submit a financial advisor query |
| POST | `/index-document` | Index a document (text, title, type) |
| DELETE | `/reset-index` | Drop and recreate the collection |

**POST /query body:**
```json
{
  "query": "My client is 34 years old, moderate risk tolerance, wants to invest $500/month for retirement. What do you recommend?"
}
```

---

## Phase 2 — Adding the Guard Node

After measuring ASR with this baseline, the secured agent will introduce:
- A **Guard Node** inserted before `generate_response`
- Rule-based validation of inputs and retrieved chunks
- Hardened system prompt explicitly instructing the model to disregard injected instructions
- Structured logging of flagged inputs/outputs for audit

The Guard Node will be a separate LangGraph branch so both agents can be compared side-by-side.
