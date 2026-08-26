# OpenVoice Lab

OpenVoice Lab is an architecture-first monorepo for an Angular frontend and a
FastAPI/Python backend that will benchmark open-weight text-to-speech models.

This initial revision intentionally contains only project scaffolding, API
contracts, replaceable interfaces, documentation, and placeholders. It does
not perform synthesis, load model weights, or run benchmarks yet.

## Repository layout

```text
frontend/  Angular application shell and feature boundaries
backend/   FastAPI application shell and backend domain boundaries
docs/      Architecture, module ownership, and API documentation
```

The primary design constraint is that inference technology remains a backend
implementation detail. Angular communicates only through HTTP contracts, API
controllers stay thin, and `SynthesisService` owns workflow orchestration
behind the replaceable `TTSInferenceAdapter` abstraction.

## Local scaffolding commands

Frontend:

```powershell
cd frontend
npm install
npm start
```

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

These commands prepare the development shells. Synthesis and benchmark routes
return `501 Not Implemented` until their application services are implemented.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Module responsibilities](docs/MODULES.md)
- [API surface](docs/API.md)

