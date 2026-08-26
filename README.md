# OpenVoice Lab

**Architecture first. Benchmarks before claims.**

OpenVoice Lab is a modular platform for evaluating and serving open-weight TTS
models.

> “I converted the architecture into a working FastAPI service while keeping
> inference implementation replaceable.”

## Stage 1 — Executable backend contract

| | |
| --- | --- |
| **Problem** | Evaluate and serve open-weight TTS models. |
| **Primary flow** | Text → model → audio + performance metrics. |
| **Target stack** | Angular, FastAPI, Python, ONNX, Kokoro, Docker. |
| **Current reality** | The API contract runs and is tested. Synthesis is deterministic mock data; no model is loaded and no audio is generated. |

```text
HTTP request
  ↓
FastAPI controller
  ↓
Pydantic validation
  ↓
SynthesisService / ModelRegistry
  ↓
Typed JSON response
```

Controllers only translate HTTP, accept validated schemas, call a service, and
return its response. Inference remains behind `TTSInferenceAdapter`; ONNX and
Kokoro do not appear in controller code.

## Run the evidence

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\uvicorn.exe app.main:app --reload
```

```text
curl http://localhost:8000/health
  ↓
{"status":"healthy"}

curl http://localhost:8000/api/models
  ↓
[{"id":"kokoro","displayName":"Kokoro","voices":["af_heart"],"variants":["fp32","quantized"]}]
```

```text
curl -X POST http://localhost:8000/api/synthesis \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","modelId":"kokoro","voiceId":"af_heart","variant":"fp32"}'
  ↓
{"status":"mock","model":"kokoro-fp32","text":"Hello world","audioUrl":null}
```

Malformed synthesis bodies return FastAPI/Pydantic `422` validation responses.
Run the contract suite with `.\.venv\Scripts\pytest.exe` from `backend/`.

## Repository

```text
frontend/   Angular shell and feature boundaries
backend/    FastAPI API, schemas, services, and runtime boundaries
docs/       Architecture, ownership, API, and delivery roadmap
```

- [Architecture](docs/ARCHITECTURE.md)
- [Module responsibilities](docs/MODULES.md)
- [API contract](docs/API.md)
- [Iterative coding map](docs/ITERATIVE_CODING_MAP.md)

**Portfolio status:** demonstrates API design, FastAPI, Pydantic validation,
service separation, and automated contract tests. Real inference remains a
later, separately evidenced stage.
