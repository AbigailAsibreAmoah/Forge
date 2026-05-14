"""
Forge Backend — FastAPI + Groq streaming
v4.0 — adds conversation coaching + browser-extension endpoints
"""
import os
import json
import httpx
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Forge API", version="4.0")

# CORS — wide open so the browser extension can call from any AI chat origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


# ---------- MODELS ----------

class PromptRequest(BaseModel):
    prompt: str

class SqlRequest(BaseModel):
    description: str
    dialect: str = "PostgreSQL"

class QuickRefineRequest(BaseModel):
    prompt: str
    context: Optional[str] = None  # site host / model name (chatgpt, claude…)

class ConvMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class FollowUpRequest(BaseModel):
    messages: List[ConvMessage]
    goal: Optional[str] = None  # what the user is ultimately trying to do


# ---------- HELPERS ----------

def groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_groq(system: str, user: str, max_tokens: int = 1024):
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", GROQ_URL, headers=groq_headers(), json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise HTTPException(status_code=resp.status_code, detail=body.decode())
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
                text = evt.get("choices", [{}])[0].get("delta", {}).get("content")
                if text:
                    yield text


async def call_groq(system: str, user: str, max_tokens: int = 256) -> str:
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(GROQ_URL, headers=groq_headers(), json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()["choices"][0]["message"]["content"].strip()


def _strip_fence(text: str) -> str:
    text = text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return text


# ---------- ROUTES ----------

@app.post("/api/forge-prompt")
async def forge_prompt(req: PromptRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    forge_system = (
        "You are an expert prompt engineer. "
        "When given a raw prompt, rewrite it to be maximally effective for a large language model: "
        "add an expert role framing, set a clear tone and target audience, constrain the output format, "
        "and require a concrete takeaway. Keep it concise — under 280 words. "
        "Return ONLY the improved prompt text, nothing else."
    )

    scores_system = (
        "You are a prompt quality evaluator. "
        "Given a prompt, return ONLY a JSON object with keys: clarity, specificity, tone, overall. "
        "Each value is an integer from 0-100. No explanation, no markdown — raw JSON only."
    )

    async def event_stream():
        try:
            async for token in stream_groq(forge_system, req.prompt):
                yield sse({"type": "token", "text": token})
            raw = _strip_fence(await call_groq(scores_system, req.prompt, max_tokens=128))
            scores = json.loads(raw)
            yield sse({"type": "scores", **scores})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/refine-quick")
async def refine_quick(req: QuickRefineRequest):
    """Non-streaming, single-shot refinement for the browser extension."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    ctx = f" The user is chatting with {req.context}." if req.context else ""
    system = (
        "You are an expert prompt engineer." + ctx +
        " Rewrite the user's prompt so an AI assistant produces a stronger, more specific answer. "
        "Add role framing, audience, format constraints, and what a great answer looks like. "
        "Keep it under 200 words. Return ONLY the rewritten prompt — no preamble, no quotes, no markdown."
    )
    refined = await call_groq(system, req.prompt, max_tokens=512)
    return {"refined": refined}


@app.post("/api/follow-ups")
async def follow_ups(req: FollowUpRequest):
    """
    Recommend 3 strong follow-up prompts based on the ongoing conversation.
    Conversation history is sent by the client per request — the server stores nothing.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")

    transcript = "\n".join(
        f"{m.role.upper()}: {m.content.strip()}" for m in req.messages[-12:]
    )
    goal_line = f"\nUSER GOAL: {req.goal.strip()}\n" if req.goal else ""

    system = (
        "You are a conversation strategist for AI chats. "
        "Read the transcript and suggest THREE high-leverage follow-up prompts the user could send next "
        "to push the conversation toward their goal. Each follow-up should be specific, ready to paste, "
        "and unlock new value (deepen, challenge, apply, or pivot). "
        "Return ONLY a JSON object: "
        '{"suggestions":[{"label":"short title","prompt":"the full prompt","why":"one sentence"}, ...]}'
    )
    user_msg = f"{goal_line}TRANSCRIPT:\n{transcript}"
    raw = _strip_fence(await call_groq(system, user_msg, max_tokens=600))
    try:
        return json.loads(raw)
    except Exception:
        return {"suggestions": [], "raw": raw}


@app.post("/api/generate-sql")
async def generate_sql(req: SqlRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    system = (
        f"You are an expert {req.dialect} SQL engineer. "
        "Given a plain-English description, write clean, optimized, production-ready SQL. "
        "Add short inline comments. Use proper indentation and CTEs where helpful. "
        "Return ONLY the SQL — no markdown fences, no prose outside comments."
    )

    async def event_stream():
        try:
            async for token in stream_groq(system, req.description):
                yield sse({"type": "token", "text": token})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL, "api_key_set": bool(GROQ_API_KEY), "version": "4.0"}


# Serve the packaged extension zip if present (built by build_extension.sh)
@app.get("/forge-extension.zip")
async def get_extension():
    path = os.path.join(os.path.dirname(__file__), "static", "forge-extension.zip")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Extension zip not built yet")
    return FileResponse(path, media_type="application/zip", filename="forge-extension.zip")


# Static files — must be mounted last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
