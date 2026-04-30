"""
Forge Backend — FastAPI + Groq streaming
Primary: llama-3.1-8b-instant  |  Fallback: llama-3.3-70b-versatile
"""
import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Forge API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
MODEL_FAST    = "llama-3.1-8b-instant"
MODEL_FALLBACK = "llama-3.3-70b-versatile"

# Minimum token count before we consider the fast model's response "good enough"
MIN_TOKENS_OK = 30


# ---------- REQUEST MODELS ----------

class PromptRequest(BaseModel):
    prompt: str
    iterate: str | None = None

class SqlRequest(BaseModel):
    description: str
    dialect: str = "PostgreSQL"


# ---------- HELPERS ----------

def groq_headers():
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_groq(system: str, user: str, model: str):
    """
    Yield (token_text, token_count) tuples from Groq SSE stream.
    Raises httpx.HTTPStatusError on non-200.
    """
    payload = {
        "model": model,
        "max_tokens": 1024,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST", GROQ_URL, headers=groq_headers(), json=payload
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise httpx.HTTPStatusError(
                    body.decode(), request=resp.request, response=resp
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    evt = json.loads(raw)
                except Exception:
                    continue
                text = evt.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if text:
                    yield text


async def stream_with_fallback(system: str, user: str):
    """
    Try fast model first. If it errors OR produces too few tokens,
    transparently switch to fallback and stream from there.
    Yields (text, model_used) tuples.
    """
    collected = []
    failed = False

    # --- attempt fast model ---
    try:
        async for token in stream_groq(system, user, MODEL_FAST):
            collected.append(token)
            yield token, MODEL_FAST
    except Exception:
        failed = True

    total_chars = sum(len(t) for t in collected)
    too_short = total_chars < MIN_TOKENS_OK * 3  # rough char estimate

    if failed or too_short:
        # fallback: stream the full response from 70b
        try:
            async for token in stream_groq(system, user, MODEL_FALLBACK):
                yield token, MODEL_FALLBACK
        except Exception as e:
            raise RuntimeError(f"Both models failed: {e}")


async def call_groq(system: str, user: str, max_tokens: int = 256) -> str:
    """Non-streaming call — tries fast model, falls back if needed."""
    for model in (MODEL_FAST, MODEL_FALLBACK):
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(GROQ_URL, headers=groq_headers(), json=payload)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                if content and len(content.strip()) > 10:
                    return content
        except Exception:
            if model == MODEL_FALLBACK:
                raise
            continue  # try fallback
    raise RuntimeError("All models failed")


# ---------- ROUTES ----------

@app.post("/api/forge-prompt")
async def forge_prompt(req: PromptRequest):
    """
    Stream an improved prompt then emit quality scores.
    SSE: { type: "token"|"model"|"scores"|"done"|"error", ... }
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    if req.iterate:
        improve_system = (
            "You are an expert prompt engineer. "
            f"You previously improved a prompt. Now refine it further based on this instruction: '{req.iterate}'. "
            "Return ONLY the refined prompt text, nothing else."
        )
    else:
        improve_system = (
            "You are an expert prompt engineer. "
            "Rewrite the given prompt to be maximally effective for a large language model: "
            "add an expert role framing, set a clear tone and target audience, constrain the output format, "
            "and require a concrete takeaway. Keep it concise — under 280 words. "
            "Return ONLY the improved prompt text, nothing else."
        )
    scores_system = (
        "You are a prompt quality evaluator. "
        "Given a prompt, return ONLY a JSON object with integer keys: clarity, specificity, tone, overall (each 0-100). "
        "No explanation, no markdown fences — raw JSON only."
    )

    async def event_stream():
        try:
            active_model = None
            async for token, model in stream_with_fallback(improve_system, req.prompt):
                if model != active_model:
                    active_model = model
                    yield sse({"type": "model", "model": model})
                yield sse({"type": "token", "text": token})

            # scores via fast→fallback non-streaming call
            raw_json = await call_groq(scores_system, req.prompt, max_tokens=128)
            raw_json = raw_json.strip().strip("`")
            if raw_json.lower().startswith("json"):
                raw_json = raw_json[4:].strip()
            scores = json.loads(raw_json)
            yield sse({"type": "scores", **scores})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/generate-sql")
async def generate_sql(req: SqlRequest):
    """
    Stream SQL from a natural-language description.
    SSE: { type: "token"|"model"|"done"|"error", ... }
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    system = (
        f"You are an expert {req.dialect} SQL engineer. "
        "Convert the plain-English description into clean, optimized, production-ready SQL. "
        "Add short inline comments on each major clause. "
        "Use proper indentation and CTEs where they improve readability. "
        "Return ONLY the SQL — no markdown fences, no explanation outside of SQL comments."
    )

    async def event_stream():
        try:
            active_model = None
            async for token, model in stream_with_fallback(system, req.description):
                if model != active_model:
                    active_model = model
                    yield sse({"type": "model", "model": model})
                yield sse({"type": "token", "text": token})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "primary_model": MODEL_FAST,
        "fallback_model": MODEL_FALLBACK,
        "api_key_set": bool(GROQ_API_KEY),
    }


# Static files last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ── NEW: Run a prompt and stream the model's actual response ──
class RunPromptRequest(BaseModel):
    prompt: str

@app.post("/api/run-prompt")
async def run_prompt(req: RunPromptRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    system = (
        "You are a helpful, knowledgeable assistant. "
        "Respond clearly and concisely to the given prompt."
    )

    async def event_stream():
        try:
            async for token, model in stream_with_fallback(system, req.prompt):
                yield sse({"type": "token", "text": token})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── NEW: Score a prompt (used for A/B compare of original) ──
class ScoreRequest(BaseModel):
    prompt: str

@app.post("/api/score-prompt")
async def score_prompt(req: ScoreRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    scores_system = (
        "You are a prompt quality evaluator. "
        "Given a prompt, return ONLY a JSON object with integer keys: clarity, specificity, tone, overall (each 0-100). "
        "No explanation, no markdown fences — raw JSON only."
    )
    try:
        raw = await call_groq(scores_system, req.prompt, max_tokens=128)
        raw = raw.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        scores = json.loads(raw)
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── NEW: Forge prompt with optional iterate instruction ──
# (Override the existing forge-prompt to support `iterate` field)
