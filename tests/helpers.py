"""Utilidades compartidas por los tests de la integración con NVIDIA NIM."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock


def make_client(*, return_value: Any = None, side_effect: Any = None) -> MagicMock:
    """Cliente OpenAI falso: nunca realiza llamadas de red."""
    client = MagicMock()
    create = client.chat.completions.create
    if side_effect is not None:
        create.side_effect = side_effect
    else:
        create.return_value = return_value
    return client


def completion(content: str | None) -> SimpleNamespace:
    """Simula una respuesta completa (no streaming) del modelo."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def completion_without_choices() -> SimpleNamespace:
    """Simula una respuesta del modelo sin ningún `choice`."""
    return SimpleNamespace(choices=[])


def chunk(text: str | None) -> SimpleNamespace:
    """Simula un fragmento (chunk) de una respuesta en streaming."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


def stream_of(*chunks: SimpleNamespace) -> Iterator[SimpleNamespace]:
    """Stream completo que entrega los fragmentos indicados."""
    return iter(chunks)


def failing_stream(exc: Exception, *chunks: SimpleNamespace) -> Iterator[SimpleNamespace]:
    """Stream que falla después de entregar los fragmentos indicados."""

    def generator() -> Iterator[SimpleNamespace]:
        yield from chunks
        raise exc

    return generator()


def http_request(
    url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
) -> SimpleNamespace:
    """Request mínimo: el SDK solo guarda la referencia en la excepción."""
    return SimpleNamespace(method="POST", url=url)


def http_response(status_code: int = 429) -> SimpleNamespace:
    """Response mínimo: el SDK solo lee `status_code`, `headers` y `request`."""
    return SimpleNamespace(
        status_code=status_code,
        headers={"x-request-id": "test-request-id"},
        request=http_request(),
    )
