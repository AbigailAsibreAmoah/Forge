"""
Forge Backend — FastAPI + Groq streaming
"""
import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Forge API", version="3.0")

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


# ---------- HELPERS ----------

def groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_groq(system: str, user: str):
    """Yield raw text tokens from Groq's OpenAI-compatible streaming API."""
    payload = {
        "model": MODEL,
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
    """Single non-streaming Groq call, returns full response text."""
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


# ---------- ROUTES ----------

@app.post("/api/forge-prompt")
async def forge_prompt(req: PromptRequest):
    """
    Stream an improved prompt, then emit quality scores.
    SSE events:
      { type: "token",  text: "..." }
      { type: "scores", clarity, specificity, tone, overall }
      { type: "done" }
      { type: "error",  message: "..." }
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    # --- AI-powered prompt type classifier ---
    # Ask Groq to classify the prompt into one of 5 categories,
    # then select the matching system prompt. No hardcoded keywords.
    classifier_system = (
        "Classify the following prompt into exactly ONE of these categories: "
        "technical, creative, analytical, conversational, instructional. "
        "Reply with just the single lowercase word, nothing else."
    )
    try:
        prompt_type = await call_groq(classifier_system, req.prompt, max_tokens=5)
        prompt_type = prompt_type.strip().lower()
    except Exception:
        prompt_type = "conversational"  # safe fallback

    SYSTEM_PROMPTS = {
        "technical": (
            "You are an expert software prompt engineer. "
            "Rewrite the given prompt into a precise technical specification an AI coding assistant can execute directly. "
            "Include: (1) exact deliverable, (2) tech stack, (3) key features as bullets, "
            "(4) data persistence if relevant, (5) UI/UX constraints if frontend. "
            "No marketing language. Under 200 words. Return ONLY the improved prompt."
        ),
        "creative": (
            "You are a creative writing prompt specialist. "
            "Rewrite the given prompt to unlock richer, more vivid output from an AI: "
            "set a strong narrative voice, establish mood and setting, define the desired style or genre, "
            "and specify length or format. Under 200 words. Return ONLY the improved prompt."
        ),
        "analytical": (
            "You are an expert research and analysis prompt engineer. "
            "Rewrite the given prompt to produce structured, evidence-based AI output: "
            "define the scope, specify the analytical framework or methodology, "
            "require citations or sources where relevant, and set the output format (table, report, bullets). "
            "Under 200 words. Return ONLY the improved prompt."
        ),
        "instructional": (
            "You are an expert instructional design prompt engineer. "
            "Rewrite the given prompt to produce clear, step-by-step AI output: "
            "define the target audience and skill level, specify the number of steps or sections, "
            "require examples or visuals where helpful, and set a practical takeaway. "
            "Under 200 words. Return ONLY the improved prompt."
        ),
        "conversational": (
            "You are an expert prompt engineer. "
            "Rewrite the given prompt to be maximally effective for a large language model: "
            "add an expert role framing, set a clear tone and target audience, constrain the output format, "
            "and require a concrete takeaway. Under 280 words. Return ONLY the improved prompt."
        ),
    }

    forge_system = SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["conversational"])

    scores_system = (
        "You are a prompt quality evaluator. "
        "Given a prompt, return ONLY a JSON object with keys: clarity, specificity, tone, overall. "
        "Each value is an integer from 0-100 representing quality in that dimension. "
        "No explanation, no markdown — raw JSON only."
    )

    async def event_stream():
        try:
            async for token in stream_groq(forge_system, req.prompt):
                yield sse({"type": "token", "text": token})

            # Non-streaming scores call
            raw_json = await call_groq(scores_system, req.prompt, max_tokens=128)
            raw_json = raw_json.strip("`").strip()
            if raw_json.startswith("json"):
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
    Stream SQL generated from a natural-language description.
    SSE events:
      { type: "token", text: "..." }
      { type: "done" }
      { type: "error", message: "..." }
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    system = (
        f"You are an expert {req.dialect} SQL engineer. "
        "Given a plain-English description, write clean, optimized, production-ready SQL. "
        "Add short inline comments explaining each major clause. "
        "Use proper indentation. Use CTEs where they improve readability. "
        "Return ONLY the SQL query — no markdown fences, no explanation text outside comments."
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
    return {"status": "ok", "model": MODEL, "api_key_set": bool(GROQ_API_KEY)}


# Static files — must be mounted last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
