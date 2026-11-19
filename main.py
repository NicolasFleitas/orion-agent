"""ORION - conexión a Nemotron con manejo de errores y streaming."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from schemas.orion_response import OrionResponse

load_dotenv()

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
TEMPERATURE = 0.3
MAX_TOKENS = 2000
TIMEOUT = 30.0

BASE_URL = os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL
MODEL = os.environ.get("NVIDIA_MODEL") or DEFAULT_MODEL

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT = (BASE_DIR / "prompts" / "system.txt").read_text().strip()
USER_PROMPT = "ORION, explica el uso de un agente de IA"


def ask_complete(client: OpenAI) -> int:
    """Solicita una respuesta completa estructurada y validada."""
    try:
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
    except APITimeoutError:
        print("Error: tiempo de espera agotado al conectar con el modelo (timeout).", file=sys.stderr)
        return 1
    except RateLimitError:
        print("Error: límite de peticiones alcanzado (rate limit). Intente más tarde.", file=sys.stderr)
        return 1
    except (APIConnectionError, APIError) as exc:
        print(f"Error en la llamada a la API: {exc}", file=sys.stderr)
        return 1

    raw = completion.choices[0].message.content if completion.choices else ""
    raw = (raw or "").strip()
    if not raw:
        print("Error: ORION devolvió una respuesta vacía.", file=sys.stderr)
        return 1

    print(f"\nRaw JSON:\n{raw}")

    try:
        data = OrionResponse.model_validate_json(raw)
    except ValidationError as exc:
        print(f"\nError de validación del schema:\n{exc}", file=sys.stderr)
        return 1

    print("\nORION (validado):")
    print(f"  nivel:          {data.nivel}")
    print(f"  palabras_clave: {', '.join(data.palabras_clave)}")
    print(f"\nRespuesta:\n{data.respuesta}")
    return 0


def ask_stream(client: OpenAI) -> int:
    """Solicita una respuesta en streaming y la valida contra el schema."""
    collected: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={"guided_json": OrionResponse.model_json_schema()},
            stream=True,
        )
        print("\nORION (streaming):")
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                collected.append(text)
        print()
    except APITimeoutError:
        print("\nError: tiempo de espera agotado al conectar con el modelo (timeout).", file=sys.stderr)
        return 1
    except RateLimitError:
        print("\nError: límite de peticiones alcanzado (rate limit). Intente más tarde.", file=sys.stderr)
        return 1
    except (APIConnectionError, APIError) as exc:
        print(f"\nError en la llamada a la API: {exc}", file=sys.stderr)
        return 1

    full_text = "".join(collected).strip()
    if not full_text:
        print("Error: ORION devolvió una respuesta vacía en streaming.", file=sys.stderr)
        return 1

    try:
        data = OrionResponse.model_validate_json(full_text)
    except ValidationError as exc:
        print(f"\nError de validación del schema en streaming:\n{exc}", file=sys.stderr)
        return 1

    print("\nORION (streaming validado):")
    print(f"  nivel:          {data.nivel}")
    print(f"  palabras_clave: {', '.join(data.palabras_clave)}")
    print(f"\nRespuesta:\n{data.respuesta}")
    return 0


def main() -> int:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print(
            "NVIDIA_API_KEY no está configurada. Copiá .env.example a .env y agregá tu clave.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=TIMEOUT)

    print(f"model:       {MODEL}")
    print(f"base_url:    {BASE_URL}")
    print(f"temperature: {TEMPERATURE}")
    print(f"max_tokens:  {MAX_TOKENS}")
    print(f"timeout:     {TIMEOUT}s")

    if "--stream" in sys.argv:
        print("conectando con NVIDIA NIM (streaming)...")
        return ask_stream(client)

    if "--complete" in sys.argv:
        print("conectando con NVIDIA NIM (respuesta completa)...")
        return ask_complete(client)

    print("conectando con NVIDIA NIM (respuesta completa)...")
    status = ask_complete(client)
    if status != 0:
        return status

    print("\nconectando con NVIDIA NIM (streaming)...")
    return ask_stream(client)


if __name__ == "__main__":
    raise SystemExit(main())
