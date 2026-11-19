"""ORION - conexión a Nemotron con respuesta estructurada validada."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from schemas.orion_response import OrionResponse

load_dotenv()

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
TEMPERATURE = 0.3
MAX_TOKENS = 2000

BASE_URL = os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL
MODEL = os.environ.get("NVIDIA_MODEL") or DEFAULT_MODEL

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT = (BASE_DIR / "prompts" / "system.txt").read_text().strip()
USER_PROMPT = "ORION, explica el uso de un agente de IA"


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
        extra_body={"guided_json": OrionResponse.model_json_schema()},
    )

    raw = completion.choices[0].message.content or ""
    if not raw:
        print("ORION devolvió una respuesta vacía.", file=sys.stderr)
        return 1

    print(f"\nRaw JSON:\n{raw}")

    try:
        data = OrionResponse.model_validate_json(raw)
    except ValidationError as exc:
        print(f"\nValidation failed:\n{exc}", file=sys.stderr)
        return 1

    print("\nORION (validado):")
    print(f"  nivel:          {data.nivel}")
    print(f"  palabras_clave: {', '.join(data.palabras_clave)}")
    print(f"\nRespuesta:\n{data.respuesta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
