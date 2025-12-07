# ⚙ server — FastAPI + Jac Bridge (MTLLM)

Backend that connects the **jac-web** UI to **Jac programs** and model endpoints.  

---

## Requirements
- **Python** ≥ 3.8
- `fastapi`, `uvicorn`
- **Jac language** runtime: `pip install jac` (used by server processes/invocations)

## Install & Run
```bash
cd server
# (optional) conda create -n jac-rpg python=3.11 -y && conda activate jac-rpg
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API: http://localhost:8000
```

### GET /api/status
Health/status of the service and current active route.

### POST /api/shutdown
Graceful stop of workers/processes.

## Project Structure
```bash
server/
├─ main.py              # FastAPI app & routes
├─ requirements.txt
|--.env
└─ README.md
```

## Notes on Jac Integration

You can keep Jac files in a top-level jac/ folder (e.g., stage1.jac, utils.jac) and import/call from server jobs.

Fine-tuning uses logged pairs (input, output, validation labels); merged weights go to 2010.

## Troubleshooting

- **CORS from frontend**: enable CORS in main.py (FastAPI middleware) or keep ports 8000 ↔ 5173.
