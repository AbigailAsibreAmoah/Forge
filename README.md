# Forge — Python Backend

FastAPI backend for Forge AI — wires **Prompt Forge** and **SQL Sorcerer** to Groq's blazing-fast LLM API.

## Stack
- **FastAPI** — async web framework
- **Groq** — LLM inference (`llama-3.3-70b-versatile`)
- **httpx** — async HTTP streaming
- **uvicorn** — ASGI server

## Quick Start (local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY=gsk_...
# Get a free key at https://console.groq.com
```

### 3. Run
```bash
python run.py
# OR
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

## Deploy to Railway

1. Push this folder to a GitHub repo
2. Create a new Railway project → **Deploy from GitHub repo**
3. Railway auto-detects the `Dockerfile`
4. Add environment variable: `GROQ_API_KEY=gsk_...`
5. Deploy — done ✓

## Project Structure
```
forge_backend/
├── main.py            ← FastAPI app + all API routes
├── run.py             ← Local dev launcher (loads .env, hot reload)
├── requirements.txt
├── Dockerfile         ← Production container for Railway
├── railway.toml       ← Railway build config
├── .env.example       ← Copy to .env and add your key
└── static/
    └── index.html     ← Forge UI
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/` | Forge UI |
| POST | `/api/forge-prompt` | Stream prompt improvement + quality scores |
| POST | `/api/generate-sql` | Stream SQL from natural language |
| GET  | `/api/health` | Health check |

### POST /api/forge-prompt
```json
{ "prompt": "Explain quantum computing simply." }
```
SSE stream:
```
data: {"type": "token", "text": "You are an expert..."}
data: {"type": "scores", "clarity": 88, "specificity": 84, "tone": 91, "overall": 90}
data: {"type": "done"}
```

### POST /api/generate-sql
```json
{ "description": "Top 10 customers by spend last 30 days", "dialect": "PostgreSQL" }
```
SSE stream:
```
data: {"type": "token", "text": "WITH recent_orders AS ("}
data: {"type": "done"}
```

## Swap the Model
Edit `MODEL` in `main.py`:
```python
MODEL = "llama-3.3-70b-versatile"   # default — fast + smart
# MODEL = "llama-3.1-8b-instant"    # faster, lighter
# MODEL = "mixtral-8x7b-32768"      # long context
```
