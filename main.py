"""
Forge Backend — FastAPI + Groq streaming
Auth: JWT (email + password)
Tiers: free (20 uses/month), paid (unlimited)
Payments: Paystack webhooks
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

PAYSTACK_SECRET_KEY   = os.getenv("PAYSTACK_SECRET_KEY", "")  # sk_live_... from Paystack dashboard

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
                ls_customer_id TEXT,                             -- Paystack customer code
                ls_sub_id      TEXT,                             -- Paystack subscription code
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT    PRIMARY KEY,           -- UUID
                user_id     INTEGER NOT NULL,
                title       TEXT,                          -- auto-generated from first prompt
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                user_id         INTEGER NOT NULL,
                turn_index      INTEGER NOT NULL,
                original_prompt TEXT    NOT NULL,
                improved_prompt TEXT,
                prompt_type     TEXT,
                score_clarity   INTEGER,
                score_specificity INTEGER,
                score_tone      INTEGER,
                score_overall   INTEGER,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
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
                        "upgrade_url": os.getenv("UPGRADE_URL", "https://paystack.com/pay/forge"),
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
    session_id: str | None = None   # existing session; None = create new

class SqlRequest(BaseModel):
    description: str
    dialect: str = "PostgreSQL"

class SessionCreateRequest(BaseModel):
    title: str | None = None

class NextPromptRequest(BaseModel):
    session_id: str

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

# ── Paystack webhook ──────────────────────────────────────────────────────────
@app.post("/api/webhooks/paystack")
async def paystack_webhook(request: Request):
    """
    Handles Paystack events: charge.success, subscription.create,
    subscription.disable, invoice.payment_failed.

    Verify the signature using PAYSTACK_SECRET_KEY set in Railway env vars.
    Paystack signs the raw body with HMAC-SHA512 and sends it in
    the X-Paystack-Signature header.
    """
    body = await request.body()

    # Verify signature
    if PAYSTACK_SECRET_KEY:
        sig = request.headers.get("x-paystack-signature", "")
        expected = hmac.new(
            PAYSTACK_SECRET_KEY.encode(), body, hashlib.sha512
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = data.get("event", "")
    obj   = data.get("data", {})

    # Paystack puts the customer email in different places depending on event
    customer_email = (
        obj.get("customer", {}).get("email", "")
        or obj.get("email", "")
    ).strip().lower()

    customer_code = obj.get("customer", {}).get("customer_code", "")
    sub_code      = obj.get("subscription_code", "") or obj.get("plan", {}).get("plan_code", "")

    with get_db() as conn:
        if event in ("charge.success", "subscription.create"):
            # Payment succeeded — upgrade to paid
            conn.execute(
                """UPDATE users
                   SET tier = 'paid', ls_customer_id = ?, ls_sub_id = ?
                   WHERE email = ?""",
                (customer_code, sub_code, customer_email)
            )

        elif event in ("subscription.disable", "invoice.payment_failed"):
            # Subscription cancelled or payment failed — downgrade
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

    # ── Session handling ──────────────────────────────────────────────────────
    session_id = req.session_id
    with get_db() as conn:
        if session_id:
            # Validate session belongs to this user
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user["id"])
            ).fetchone()
            if not row:
                session_id = None

        if not session_id:
            session_id = secrets.token_urlsafe(16)
            title = req.prompt[:60].strip()
            conn.execute(
                "INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)",
                (session_id, user["id"], title)
            )

        turn_index = conn.execute(
            "SELECT COUNT(*) FROM conversation_turns WHERE session_id = ?",
            (session_id,)
        ).fetchone()[0]

    async def event_stream():
        improved_parts = []
        scores_data = {}
        try:
            async for token in stream_groq(forge_system, req.prompt):
                improved_parts.append(token)
                yield sse({"type": "token", "text": token})

            raw_json = await call_groq(scores_system, req.prompt, max_tokens=128)
            raw_json = raw_json.strip("`").strip()
            if raw_json.startswith("json"):
                raw_json = raw_json[4:].strip()
            scores_data = json.loads(raw_json)
            yield sse({"type": "scores", **scores_data})

            # Persist the turn
            improved = "".join(improved_parts)
            with get_db() as conn:
                conn.execute(
                    """INSERT INTO conversation_turns
                       (session_id, user_id, turn_index, original_prompt, improved_prompt,
                        prompt_type, score_clarity, score_specificity, score_tone, score_overall)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (session_id, user["id"], turn_index, req.prompt, improved,
                     prompt_type,
                     scores_data.get("clarity"), scores_data.get("specificity"),
                     scores_data.get("tone"), scores_data.get("overall"))
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
                    (session_id,)
                )

            yield sse({"type": "session", "session_id": session_id, "turn_index": turn_index})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Session & conversation history routes ─────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions(user: dict = Depends(current_user)):
    """Return all sessions for the current user, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT s.id, s.title, s.created_at, s.updated_at,
                      COUNT(t.id) as turn_count
               FROM sessions s
               LEFT JOIN conversation_turns t ON t.session_id = s.id
               WHERE s.user_id = ?
               GROUP BY s.id
               ORDER BY s.updated_at DESC
               LIMIT 50""",
            (user["id"],)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(current_user)):
    """Return all turns in a session."""
    with get_db() as conn:
        sess = conn.execute(
            "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, user["id"])
        ).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        turns = conn.execute(
            """SELECT * FROM conversation_turns
               WHERE session_id = ? ORDER BY turn_index ASC""",
            (session_id,)
        ).fetchall()

    return {
        "session": dict(sess),
        "turns": [dict(t) for t in turns],
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(current_user)):
    with get_db() as conn:
        sess = conn.execute(
            "SELECT id FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, user["id"])
        ).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        conn.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return {"deleted": True}


@app.post("/api/sessions/{session_id}/recommend")
async def recommend_next_prompt(session_id: str, user: dict = Depends(current_user)):
    """Analyse the session history and recommend the single best next prompt."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    with get_db() as conn:
        sess = conn.execute(
            "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, user["id"])
        ).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        turns = conn.execute(
            """SELECT original_prompt, improved_prompt, prompt_type,
                      score_clarity, score_specificity, score_tone, score_overall
               FROM conversation_turns WHERE session_id = ?
               ORDER BY turn_index ASC""",
            (session_id,)
        ).fetchall()

    if not turns:
        raise HTTPException(status_code=400, detail="No turns in this session yet")

    history_summary = "\n".join(
        f"Turn {i+1} [{t['prompt_type']}] (clarity:{t['score_clarity']} spec:{t['score_specificity']} "
        f"tone:{t['score_tone']} overall:{t['score_overall']})\n"
        f"  Original: {t['original_prompt'][:200]}\n"
        f"  Improved: {(t['improved_prompt'] or '')[:200]}"
        for i, t in enumerate([dict(r) for r in turns])
    )

    system = (
        "You are an expert AI prompt strategist. "
        "Analyse the conversation history below and recommend the single best NEXT prompt "
        "the user should try. Your recommendation must directly build on what has been explored, "
        "address any weak scores (clarity/specificity/tone), and push toward best practice, "
        "efficiency, productivity, and accuracy. "
        "Respond ONLY with a JSON object with these keys: "
        '{"recommendation": "<the exact next prompt text ready to paste>", '
        '"rationale": "<2-3 sentence explanation of why this is the ideal next step>", '
        '"focus_area": "<one of: clarity | specificity | tone | depth | efficiency | accuracy>", '
        '"predicted_improvement": "<short phrase, e.g. +18 overall score>"}'
        " No markdown, no extra keys."
    )

    raw = await call_groq(system, history_summary, max_tokens=512)
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        result = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Model returned unparseable JSON")

    return result


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
                "upgrade_url": os.getenv("UPGRADE_URL", "https://paystack.com/pay/forge"),
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


# ── Browser Extension endpoints ───────────────────────────────────────────────
# These are called from the Chrome extension content script.
# Auth is via Bearer token stored in extension storage (same JWT as web app).

class ExtTurnRequest(BaseModel):
    """One user→AI exchange scraped from a supported AI chat UI."""
    ai_platform:   str          # "chatgpt" | "claude" | "gemini" | "perplexity" | "other"
    session_id:    str | None = None
    user_message:  str
    ai_response:   str | None = None   # may be empty if scraped before AI finishes
    turn_index:    int = 0

class ExtRecommendRequest(BaseModel):
    """Raw turn list from the extension — no DB session required."""
    ai_platform:  str
    turns: list[dict]           # [{user_message, ai_response, turn_index}]


@app.post("/api/ext/track-turn")
async def ext_track_turn(req: ExtTurnRequest, user: dict = Depends(current_user)):
    """
    Called by the browser extension each time a new user→AI exchange completes.
    Persists the turn and returns a lightweight quality score + session_id.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    # Resolve / create session
    with get_db() as conn:
        session_id = req.session_id
        if session_id:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user["id"])
            ).fetchone()
            if not row:
                session_id = None

        if not session_id:
            session_id = secrets.token_urlsafe(16)
            title = f"[{req.ai_platform.upper()}] {req.user_message[:55].strip()}"
            conn.execute(
                "INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)",
                (session_id, user["id"], title)
            )

        turn_index = conn.execute(
            "SELECT COUNT(*) FROM conversation_turns WHERE session_id = ?",
            (session_id,)
        ).fetchone()[0]

    # Score the user message
    scores_system = (
        "You are a prompt quality evaluator. "
        "Given a user message sent to an AI, return ONLY a JSON object with keys: "
        "clarity, specificity, tone, overall — each an integer 0-100. "
        "No explanation, no markdown, raw JSON only."
    )
    try:
        raw_json = await call_groq(scores_system, req.user_message, max_tokens=80)
        raw_json = raw_json.strip().strip("`")
        if raw_json.startswith("json"):
            raw_json = raw_json[4:].strip()
        scores = json.loads(raw_json)
    except Exception:
        scores = {"clarity": 0, "specificity": 0, "tone": 0, "overall": 0}

    # Persist turn
    with get_db() as conn:
        conn.execute(
            """INSERT INTO conversation_turns
               (session_id, user_id, turn_index, original_prompt, improved_prompt,
                prompt_type, score_clarity, score_specificity, score_tone, score_overall)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, user["id"], turn_index,
             req.user_message,
             req.ai_response,          # store AI response in improved_prompt field
             req.ai_platform,
             scores.get("clarity"), scores.get("specificity"),
             scores.get("tone"), scores.get("overall"))
        )
        conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,)
        )

    return {
        "session_id": session_id,
        "turn_index": turn_index,
        "scores": scores,
    }


@app.post("/api/ext/recommend")
async def ext_recommend(req: ExtRecommendRequest, user: dict = Depends(current_user)):
    """
    Takes raw turns from the extension (no DB required) and returns the
    single best next prompt recommendation.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    if not req.turns:
        raise HTTPException(status_code=400, detail="No turns provided")

    history_text = "\n\n".join(
        f"Turn {t.get('turn_index', i)+1}:\n"
        f"  User: {str(t.get('user_message',''))[:300]}\n"
        f"  AI:   {str(t.get('ai_response',''))[:300]}"
        for i, t in enumerate(req.turns)
    )

    system = (
        f"You are an expert AI prompt strategist. The user is chatting with {req.ai_platform}. "
        "Analyse the conversation below and recommend the single best NEXT prompt they should send. "
        "It must directly build on what has been covered, be clearer and more specific, "
        "and push toward best practice, efficiency, productivity, and accuracy. "
        "Respond ONLY with a JSON object: "
        '{"recommendation": "<exact next prompt, ready to paste>", '
        '"rationale": "<2-3 sentences on why this is the ideal next step>", '
        '"focus_area": "<clarity|specificity|depth|efficiency|accuracy|follow-up>", '
        '"predicted_improvement": "<short phrase>"} '
        "No markdown, no extra keys."
    )

    raw = await call_groq(system, history_text, max_tokens=512)
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        result = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Model returned unparseable JSON")

    return result


# Static files — mounted last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)