"""
Forge Backend — FastAPI + Groq streaming
Auth: JWT (email + password)
Tiers: free (20 uses/month), paid (unlimited)
Payments: Lemon Squeezy webhooks
"""
import os
import json
import httpx
import sqlite3
import hashlib
import hmac
import secrets
import time
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import jwt  # PyJWT
from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

app = FastAPI(title="Forge API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Env ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
GROQ_URL              = "https://api.groq.com/openai/v1/chat/completions"
MODEL                 = "llama-3.3-70b-versatile"

JWT_SECRET            = os.getenv("JWT_SECRET", secrets.token_hex(32))  # set a stable value in Railway
JWT_EXPIRY_DAYS       = 30

LEMON_WEBHOOK_SECRET  = os.getenv("LEMON_WEBHOOK_SECRET", "")  # from Lemon Squeezy dashboard

FREE_MONTHLY_LIMIT    = 20   # refinements per month for free tier

DB_PATH               = os.getenv("DB_PATH", "/data/forge.db")  # use Railway volume or local path

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                tier          TEXT    NOT NULL DEFAULT 'free',   -- 'free' | 'paid'
                usage_count   INTEGER NOT NULL DEFAULT 0,
                usage_reset   TEXT    NOT NULL,                  -- ISO date of next monthly reset
                ls_customer_id TEXT,                             -- Lemon Squeezy customer id
                ls_sub_id      TEXT,                             -- Lemon Squeezy subscription id
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

init_db()

# ── Auth helpers ──────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{h}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split(":", 1)
        return hmac.compare_digest(
            hashlib.sha256((salt + password).encode()).hexdigest(), h
        )
    except Exception:
        return False

def make_token(user_id: int, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired — please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def next_reset_date() -> str:
    """First day of next month, ISO format."""
    now = datetime.now(timezone.utc)
    if now.month == 12:
        return datetime(now.year + 1, 1, 1).isoformat()
    return datetime(now.year, now.month + 1, 1).isoformat()

# ── Auth dependency ───────────────────────────────────────────────────────────
def current_user(authorization: str = Header(default="")):
    """FastAPI dependency — extracts and validates the Bearer token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization[7:]
    payload = decode_token(token)
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (payload["sub"],)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(row)

# ── Usage gate ────────────────────────────────────────────────────────────────
def check_and_increment_usage(user: dict):
    """
    Raises 429 if a free user has hit their monthly limit.
    Resets the counter if the reset date has passed.
    Increments usage for free users (paid users are unlimited).
    """
    with get_db() as conn:
        # Refresh from DB to avoid stale data
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        row = dict(row)

        # Roll over usage if past reset date
        now_iso = datetime.now(timezone.utc).isoformat()
        if now_iso >= row["usage_reset"]:
            conn.execute(
                "UPDATE users SET usage_count = 0, usage_reset = ? WHERE id = ?",
                (next_reset_date(), row["id"])
            )
            row["usage_count"] = 0

        if row["tier"] == "free":
            if row["usage_count"] >= FREE_MONTHLY_LIMIT:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "code": "limit_reached",
                        "message": f"You've used all {FREE_MONTHLY_LIMIT} free refinements this month.",
                        "upgrade_url": os.getenv("UPGRADE_URL", "https://forge.lemonsqueezy.com/checkout"),
                    }
                )
            conn.execute(
                "UPDATE users SET usage_count = usage_count + 1 WHERE id = ?",
                (row["id"],)
            )

# ── Pydantic models ───────────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class PromptRequest(BaseModel):
    prompt: str

class SqlRequest(BaseModel):
    description: str
    dialect: str = "PostgreSQL"

# ── Groq helpers ──────────────────────────────────────────────────────────────
def groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

async def stream_groq(system: str, user: str):
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

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.post("/api/auth/signup")
async def signup(req: SignupRequest):
    email = req.email.strip().lower()
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists")

        conn.execute(
            "INSERT INTO users (email, password_hash, usage_reset) VALUES (?, ?, ?)",
            (email, hash_password(req.password), next_reset_date())
        )
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    token = make_token(user_id, email)
    return {
        "token": token,
        "user": {"email": email, "tier": "free", "usage_count": 0, "limit": FREE_MONTHLY_LIMIT}
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    email = req.email.strip().lower()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    row = dict(row)
    token = make_token(row["id"], row["email"])
    return {
        "token": token,
        "user": {
            "email": row["email"],
            "tier": row["tier"],
            "usage_count": row["usage_count"],
            "limit": FREE_MONTHLY_LIMIT if row["tier"] == "free" else None,
        }
    }


@app.get("/api/auth/me")
async def me(user: dict = Depends(current_user)):
    return {
        "email": user["email"],
        "tier": user["tier"],
        "usage_count": user["usage_count"],
        "limit": FREE_MONTHLY_LIMIT if user["tier"] == "free" else None,
        "usage_reset": user["usage_reset"],
    }

# ── Lemon Squeezy webhook ─────────────────────────────────────────────────────
@app.post("/api/webhooks/lemonsqueezy")
async def lemonsqueezy_webhook(request: Request):
    """
    Handles subscription_created, subscription_updated, subscription_cancelled.
    Verify the signature with LEMON_WEBHOOK_SECRET set in Railway env vars.
    """
    body = await request.body()

    # Verify signature
    if LEMON_WEBHOOK_SECRET:
        sig = request.headers.get("X-Signature", "")
        expected = hmac.new(
            LEMON_WEBHOOK_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = data.get("meta", {}).get("event_name", "")
    attrs  = data.get("data", {}).get("attributes", {})
    customer_email = attrs.get("user_email", "").strip().lower()
    customer_id    = str(data.get("data", {}).get("relationships", {}).get("customer", {}).get("data", {}).get("id", ""))
    sub_id         = str(data.get("data", {}).get("id", ""))
    status         = attrs.get("status", "")

    with get_db() as conn:
        if event in ("subscription_created", "subscription_updated"):
            new_tier = "paid" if status == "active" else "free"
            conn.execute(
                """UPDATE users
                   SET tier = ?, ls_customer_id = ?, ls_sub_id = ?
                   WHERE email = ?""",
                (new_tier, customer_id, sub_id, customer_email)
            )

        elif event == "subscription_cancelled":
            # Downgrade at period end — Lemon sends a final "subscription_updated"
            # with status="cancelled" or "expired" when it actually ends.
            conn.execute(
                "UPDATE users SET tier = 'free' WHERE email = ?",
                (customer_email,)
            )

    return {"received": True}

# ── API routes ────────────────────────────────────────────────────────────────
@app.post("/api/forge-prompt")
async def forge_prompt(req: PromptRequest, user: dict = Depends(current_user)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    check_and_increment_usage(user)

    classifier_system = (
        "Classify the following prompt into exactly ONE of these categories: "
        "technical, creative, analytical, conversational, instructional. "
        "Reply with just the single lowercase word, nothing else."
    )
    try:
        prompt_type = (await call_groq(classifier_system, req.prompt, max_tokens=5)).lower()
    except Exception:
        prompt_type = "conversational"

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
async def generate_sql(req: SqlRequest, user: dict = Depends(current_user)):
    """SQL generation — paid users only."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    if user["tier"] != "paid":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "paid_only",
                "message": "SQL generation is a paid feature.",
                "upgrade_url": os.getenv("UPGRADE_URL", "https://forge.lemonsqueezy.com/checkout"),
            }
        )

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


# Static files — mounted last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)