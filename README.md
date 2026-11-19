# ORION — Asistente de IA con NVIDIA NIM + Nemotron

Backend en Python que conecta a ORION con un modelo **Nemotron** a través de
**NVIDIA NIM**. El prompt define el rol y los límites del asistente, la respuesta
se solicita como JSON y se **valida con un schema Pydantic** antes de usarse.
Incluye manejo explícito de timeouts, respuestas vacías y rate limits, salida
completa y streaming, y tests con mocks que no llaman a la API real.

## Requisitos

- Python 3.10 o superior
- Una API key de NVIDIA NIM ([build.nvidia.com](https://build.nvidia.com))

## Instalación

```bash
git clone https://github.com/NicolasFleitas/orion-agent
cd orion-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # para los tests: requirements-dev.txt
```

## Configuración

```bash
cp .env.example .env
```

Completá tu clave en `.env`:

| Variable | Obligatoria | Valor por defecto | Descripción |
| --- | --- | --- | --- |
| `NVIDIA_API_KEY` | Sí | — | Clave de NVIDIA NIM |
| `NVIDIA_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | Endpoint del API |
| `NVIDIA_MODEL` | No | `nvidia/nemotron-3.5-lightning-30b-a3b` | Modelo Nemotron a usar |

`.env` está en `.gitignore`, así que la clave nunca se commitea.

## Ejecución

```bash
python main.py             # respuesta completa y luego streaming
python main.py --complete  # solo respuesta completa
python main.py --stream    # solo streaming
```

Cada corrida imprime el modelo, la configuración de generación y la respuesta
validada (`respuesta`, `nivel`, `palabras_clave`). Ante un fallo, el proceso
termina con código 1 y un mensaje en `stderr`. Los parámetros `temperature`,
`max_tokens` y `timeout` se configuran en `main.py`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Los tests usan un cliente OpenAI falso (`unittest.mock`): no necesitan
credenciales ni acceso a la red. Cubren respuesta exitosa, respuesta vacía,
timeout, rate limit, JSON inválido o fuera de schema, y streaming.

## Estructura

```text
main.py                     # punto de entrada: respuesta completa y streaming
prompts/system.txt          # rol, límites y formato JSON de ORION
schemas/orion_response.py   # schema Pydantic de la respuesta
tests/                      # tests con mocks (sin red)
```

## Manejo de errores

| Caso | Comportamiento |
| --- | --- |
| Timeout | Mensaje controlado y salida con código 1 |
| Respuesta vacía | Se detecta antes de validar y se informa |
| Rate limit | Mensaje controlado y salida con código 1 |
| JSON inválido | Se rechaza con el detalle de validación del schema |
| Falta `NVIDIA_API_KEY` | Se avisa cómo configurarla y se corta la ejecución |
