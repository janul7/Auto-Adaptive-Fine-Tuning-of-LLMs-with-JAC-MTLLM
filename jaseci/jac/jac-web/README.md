# 🌐 jac-web — Frontend (React + Vite)

User interface for *Auto-Adaptive RPG Generator*. It lets you:
- pick an *application* (🕹 rpg game or other tasks like summarizer.jac, paper_revision.jac),
- view model outputs streamed back from the backend.

The backend (FastAPI) talks to Jac programs and model ports (base/fine-tuned/fallback).

---

## Requirements
- *Node.js* ≥ 18
- npm (or pnpm/yarn)

## Quick Start
```bash
cd jac-web
npm install
npm run dev
# UI: http://localhost:5173
```

## Configuration

The UI calls the backend (default http://localhost:8000).

Create `jac-web/.env` if you need to override:

```
VITE_API_URL=http://localhost:8000
```

## What the UI sends

POST {VITE_API_URL}/api/start with JSON:

```json
{
  "task": "summarize",                 
  "inputs": { "prompt": "..." },      
  "api_base": "None"                  
}
```

## Typical Flow (matches project sketch)

- Choose rpg game (or another task).
- Submit → UI shows outputs; background data collection + fine-tune loop happens on the server.

## Project Structure
```
jac-web/
├─ src/
│  ├─ components/       # UI widgets
│  ├─ pages/            # Views
│  ├─ lib/              # API client / helpers
│  └─ main.jsx
├─ index.html
├─ package.json
└─ README.md
```

## Scripts

- `npm run dev` — start dev server  
- `npm run build` — production build  
- `npm run preview` — preview production build  

## Troubleshooting

- **CORS / 404**: ensure backend is running on 8000 or set `VITE_API_URL`.  
- **Port busy**: stop other dev servers or change Vite port.  
- **Blank data**: check the model ports (19000/2000/2010/9010) are reachable from backend.
