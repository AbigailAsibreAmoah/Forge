#!/usr/bin/env python3
"""
One-command local launcher: loads .env then starts uvicorn with reload.
For production (Railway), the Dockerfile CMD is used instead.
"""
import os
from pathlib import Path

# Load .env if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import uvicorn  # noqa: E402

if __name__ == "__main__":
    key = os.environ.get("GROQ_API_KEY", "")
    if not key or key.startswith("gsk_your"):
        print("\n⚠  GROQ_API_KEY not set — copy .env.example to .env and add your key.\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
