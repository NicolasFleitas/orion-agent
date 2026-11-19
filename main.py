"""ORION - primera conexión a un modelo Nemotron servido por NVIDIA NIM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
TEMPERATURE = 0.3
MAX_TOKENS = 1000

BASE_URL = os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL
MODEL = os.environ.get("NVIDIA_MODEL") or DEFAULT_MODEL

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT = (BASE_DIR / "prompts" / "system.txt").read_text().strip()
USER_PROMPT = "ORION, ¿Cómo se usa una API REST?"


def main() -> int:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print(
            "NVIDIA_API_KEY no está configurada. Copiá .env.example a .env y agregá tu clave.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=30)

    print(f"model:       {MODEL}")
    print(f"base_url:    {BASE_URL}")
    print(f"temperature: {TEMPERATURE}")
    print(f"max_tokens:  {MAX_TOKENS}")
    print("conectando con NVIDIA NIM...")

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    content = (completion.choices[0].message.content or "")
    if not content:
        print("ORION devolvió una respuesta vacía.", file=sys.stderr)
        return 1

    print(f"\nORION: {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
