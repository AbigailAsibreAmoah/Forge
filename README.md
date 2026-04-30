# Forge — Python Backend

FastAPI + Groq streaming backend for the Forge AI studio.
**Primary model:** `llama-3.1-8b-instant` · **Fallback:** `llama-3.3-70b-versatile`

## Features
- **Prompt Forge** — rewrites prompts + quality scores (clarity / specificity / tone / overall)
- **Word-level diff** — side-by-side original vs improved with highlighted changes
- **Iterate** — refine the output with follow-up instructions
- **Run it** — execute the improved prompt and stream the real model response
- **A/B Compare** — scores original vs forged side-by-side, highlights the winner
- **Session History** — all runs saved in-session, click any to restore and re-run
- **SQL Sorcerer** — NL → SQL with dialect support + iterate
- **SSE streaming** — tokens stream in real time from Groq

---

## Local Development

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key (get free key at console.groq.com)
cp .env.example .env
# Edit .env → GROQ_API_KEY=gsk_...

# 3. Run
python run.py
# → http://localhost:8000
```

---

## Deployment (pick one)

### Railway (easiest — 2 minutes)
1. Push this folder to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Select your repo
4. Add env variable: GROQ_API_KEY=gsk_...
5. Done — Railway auto-detects the Dockerfile

### Render
1. Push to GitHub
2. Go to render.com → New → Web Service → connect repo
3. Render reads render.yaml automatically
4. Set GROQ_API_KEY in the Environment section
5. Deploy

### Fly.io
```bash
fly launch
fly secrets set GROQ_API_KEY=gsk_...
fly deploy
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/forge-prompt | Stream improved prompt + scores |
| POST | /api/run-prompt | Stream actual model response to a prompt |
| POST | /api/score-prompt | Score a prompt (returns JSON) |
| POST | /api/generate-sql | Stream SQL from natural language |
| GET  | /api/health | Health check |
