"""Tests con mocks de la integración de ORION con NVIDIA NIM + Nemotron.

Ningún test toca la red ni requiere credenciales: el cliente OpenAI se
reemplaza por un mock que devuelve respuestas preparadas o levanta los errores
del SDK (timeout, rate limit) de forma controlada.
"""

from __future__ import annotations

import json

import pytest
from openai import APITimeoutError, RateLimitError

import main
from schemas.orion_response import OrionResponse
from tests.helpers import (
    chunk,
    completion,
    completion_without_choices,
    failing_stream,
    http_request,
    http_response,
    make_client,
    stream_of,
)

PAYLOAD_VALIDO = {
    "respuesta": (
        "Un agente de IA percibe su entorno y actúa para alcanzar un objetivo. "
        "Usa un modelo de lenguaje como motor de razonamiento."
    ),
    "nivel": "basico",
    "palabras_clave": ["agente", "LLM", "objetivo"],
}
JSON_VALIDO = json.dumps(PAYLOAD_VALIDO, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Respuesta completa
# ---------------------------------------------------------------------------


def test_complete_devuelve_respuesta_estructurada_validada(capsys):
    client = make_client(return_value=completion(JSON_VALIDO))

    assert main.ask_complete(client) == 0

    salida = capsys.readouterr().out
    assert PAYLOAD_VALIDO["nivel"] in salida
    for palabra in PAYLOAD_VALIDO["palabras_clave"]:
        assert palabra in salida
    assert PAYLOAD_VALIDO["respuesta"] in salida


def test_complete_envia_schema_y_parametros_de_generacion():
    client = make_client(return_value=completion(JSON_VALIDO))

    main.ask_complete(client)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == main.MODEL
    assert kwargs["temperature"] == main.TEMPERATURE
    assert kwargs["max_tokens"] == main.MAX_TOKENS
    assert kwargs["extra_body"]["guided_json"] == OrionResponse.model_json_schema()
    assert [m["role"] for m in kwargs["messages"]] == ["system", "user"]
    assert kwargs["messages"][0]["content"] == main.SYSTEM_PROMPT


@pytest.mark.parametrize("contenido", [None, "", "   \n\t "])
def test_complete_detecta_respuesta_vacia(contenido, capsys):
    client = make_client(return_value=completion(contenido))

    assert main.ask_complete(client) == 1
    assert "vacía" in capsys.readouterr().err


def test_complete_detecta_respuesta_sin_choices(capsys):
    client = make_client(return_value=completion_without_choices())

    assert main.ask_complete(client) == 1
    assert "vacía" in capsys.readouterr().err


def test_complete_rechaza_json_malformado(capsys):
    client = make_client(return_value=completion("esto no es JSON"))

    assert main.ask_complete(client) == 1
    assert "validación" in capsys.readouterr().err


def test_complete_rechaza_json_fuera_de_schema(capsys):
    fuera_de_schema = json.dumps(
        {
            "respuesta": "texto",
            "nivel": "experto",  # no pertenece al Literal del schema
            "palabras_clave": [],
        }
    )
    client = make_client(return_value=completion(fuera_de_schema))

    assert main.ask_complete(client) == 1
    assert "validación" in capsys.readouterr().err


def test_complete_maneja_timeout(capsys):
    client = make_client(
        side_effect=APITimeoutError(request=http_request())
    )

    assert main.ask_complete(client) == 1
    assert "timeout" in capsys.readouterr().err.lower()


def test_complete_maneja_rate_limit(capsys):
    client = make_client(
        side_effect=RateLimitError(
            "rate limit exceeded", response=http_response(429), body=None
        )
    )

    assert main.ask_complete(client) == 1
    assert "rate limit" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_une_fragmentos_y_valida(capsys):
    fragmentos = [
        JSON_VALIDO[:10],
        JSON_VALIDO[10:40],
        JSON_VALIDO[40:],
    ]
    client = make_client(return_value=stream_of(*(chunk(t) for t in fragmentos)))

    assert main.ask_stream(client) == 0

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["extra_body"]["guided_json"] == OrionResponse.model_json_schema()

    salida = capsys.readouterr().out
    assert PAYLOAD_VALIDO["nivel"] in salida
    assert PAYLOAD_VALIDO["respuesta"] in salida


def test_stream_ignora_fragmentos_sin_contenido(capsys):
    client = make_client(
        return_value=stream_of(chunk(None), chunk(JSON_VALIDO), chunk(""))
    )

    assert main.ask_stream(client) == 0
    assert PAYLOAD_VALIDO["nivel"] in capsys.readouterr().out


def test_stream_detecta_respuesta_vacia(capsys):
    client = make_client(return_value=stream_of())

    assert main.ask_stream(client) == 1
    assert "vacía" in capsys.readouterr().err


def test_stream_maneja_timeout(capsys):
    client = make_client(
        return_value=failing_stream(
            APITimeoutError(request=http_request()), chunk('{"resp')
        )
    )

    assert main.ask_stream(client) == 1
    assert "timeout" in capsys.readouterr().err.lower()


def test_stream_maneja_rate_limit(capsys):
    client = make_client(
        return_value=failing_stream(
            RateLimitError(
                "rate limit exceeded", response=http_response(429), body=None
            ),
            chunk('{"resp'),
        )
    )

    assert main.ask_stream(client) == 1
    assert "rate limit" in capsys.readouterr().err.lower()
