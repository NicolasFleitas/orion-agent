"""Pydantic schemas for ORION responses."""

from typing import Literal

from pydantic import BaseModel, Field


class OrionResponse(BaseModel):
    """Respuesta estructurada de ORION."""

    respuesta: str = Field(description="Explicación técnica, máximo dos párrafos")
    nivel: Literal["basico", "intermedio", "avanzado"] = Field(
        description="Nivel de dificultad del tema"
    )
    palabras_clave: list[str] = Field(
        max_length=5, description="Términos técnicos relevantes"
    )
