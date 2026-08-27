# Stage 12 — Interactive SpeechT5 Experiment Interface

Status: implemented and verified on 2026-08-27

This file is the durable source of truth for Stage 12. Update the progress ledger below as work is completed so the task can resume safely after context compaction.

## Objective

Add a third top-level Angular destination at `/experiments/stage11` without changing the existing Synthesis and Benchmarks behavior. The page must present verified Stage 11 training evidence and run real, self-hosted CPU comparisons across pinned pretrained SpeechT5, V1 baseline, V2 term balance, and V3 replay.

## Locked decisions

- CPU inference only for Stage 12; no RunPod and no external inference API.
- Fixture and custom modes both perform live inference.
- Historical RTX 4090 evidence and live CPU metrics are always labeled separately.
- Training statistics are parsed from verified local artifacts, never duplicated as hardcoded UI claims.
- Locked fixtures come from the shared Stage 11 test manifest.
- Custom comparisons require explicit target terms.
- Models run sequentially on the same CPU for fair comparisons and bounded memory.
- Per-model failure does not discard successful results.
- Angular knows API fields and public model IDs only.
- Generated Stage 12 artifacts and model weights remain outside Git.

## Backend deliverables

- `app/api/experiments.py`: thin report, fixture, model, comparison, status, event, and cancellation controllers.
- `app/schemas/experiment.py`: Pydantic contracts for evidence and live comparisons.
- `app/experiments/`: artifact report, fixture catalog, immutable model registry, scorer, persistent job store, orchestration service, and speaker-profile preparation.
- `app/inference/speecht5_cpu.py`: replaceable CPU SpeechT5 inference implementation.
- Settings for Stage 11 artifacts, Stage 12 output, CPU queue/cache limits, and pinned revisions.
- Static serving for generated experiment WAV files.
- Optional pinned experiment dependencies so the Kokoro backend remains lightweight.

## Frontend deliverables

- Lazy route `/experiments/stage11` and third navigation tab `Experiment`.
- Evidence-first Stage 11 page with integrity status, result table, pipeline, dataset strategies, frozen configuration, validation-loss chart with accessible table, and provenance.
- Live lab with `Locked fixture` and `Custom text` modes.
- Searchable fixtures, explicit target-term chips, independent sanitizer/normalizer controls, model selection, progress, cancellation, retry guidance, progressive audio, ASR text, term scoring, WER, and CPU metrics.
- Responsive dark technical presentation consistent with the existing site, visible focus, 44px targets, table fallbacks, `aria-live`, and reduced-motion behavior.

## API surface

```text
GET    /api/experiments/stage11/report
GET    /api/experiments/stage11/fixtures
GET    /api/experiments/stage11/models
POST   /api/experiments/stage11/comparisons
GET    /api/experiments/stage11/comparisons/{jobId}
GET    /api/experiments/stage11/comparisons/{jobId}/events
DELETE /api/experiments/stage11/comparisons/{jobId}
```

## Artifact and provenance requirements

- Read `artifacts/stage11/full-training/final-audit.json`, per-variant provenance/training/evaluation files, dataset lock, and selected-model files.
- Fail closed when required evidence, hashes, revisions, or selected model files are invalid.
- Prepare a deterministic Stage 12 speaker profile from the same locked validation reference and pinned speaker-encoder revision because Stage 11 did not export its embedding.
- Persist live jobs atomically under `artifacts/stage12/comparisons/<job-id>/` with request, status, result, audio, and SHA-256 manifest.

## Live execution order

```text
validate request
→ resolve fixture/custom target terms
→ normalize then sanitize
→ load pinned model on CPU
→ synthesize WAV
→ expose audio immediately
→ transcribe with pinned Whisper
→ calculate exact target-term accuracy and WER
→ attach inference/load/ASR/RTF/memory metrics
→ persist verified result
```

## Required tests

- Report values match Stage 11 artifacts exactly; invalid evidence fails closed.
- Fixture catalog is backed by the shared locked manifest and supports filters/pagination.
- All request-mode and target-term validation rules.
- Exact scorer mathematics and term match details.
- Job progress, partial failure, cancellation, persistence, and restart recovery.
- Preprocessing exact text and inference-only timing boundaries.
- Model cache cold/warm behavior.
- Real CPU synthesis creates playable WAV from pretrained and all three selected models.
- Third Angular tab and deep link, loading/error/empty/progress states, both modes, independent toggles, progressive result cards, audio, and accessible chart/table rendering.
- Existing backend and frontend regression suites remain green.
- Browser acceptance at 375, 768, 1024, and 1440px with keyboard and reduced motion.

## Documentation and evidence

Preserve all earlier stages while updating README, ARCHITECTURE, MODULES, API, and ITERATIVE_CODING_MAP. Create a genuine Stage 12 screenshot and document one exercised fixture comparison and one exercised custom comparison. Never publish invented CPU measurements.

## Acceptance criteria

A fresh browser can open the third tab, inspect verified Stage 11 evidence, select a locked fixture or submit custom text plus target terms, compare at least two and up to all four SpeechT5 variants, play each successful CPU-generated WAV, and inspect ASR transcription, correct/missed terms, WER, latency, duration, RTF, memory, cold/warm state, and provenance. Existing Synthesis and Benchmarks behavior remains unchanged.

## Progress ledger

- [x] Persist approved Stage 12 plan.
- [x] Audit current app/runtime and lock artifact contracts.
- [x] Implement artifact report, fixture catalog, schemas, and tests.
- [x] Implement model registry, CPU inference, speaker profile, scorer, persistent jobs, and tests.
- [x] Compose FastAPI routes/settings/static audio and verify API behavior.
- [x] Implement lazy Angular Experiment tab and evidence presentation.
- [x] Implement live fixture/custom comparison workflow.
- [x] Run unit, integration, build, accessibility, and browser acceptance checks.
- [x] Update project documentation with genuine evidence.
- [x] Commit `feat(experiment): present Stage 11 evidence and live model comparisons`.

## Verification ledger

- Backend: 71 tests passed locally (69 portable plus two explicitly marked
  local-artifact integrations); Ruff passed across `app`, `tests`, and `scripts`.
- Frontend: 24 tests passed; Angular production build passed; the Experiment
  feature remains a lazy route (82.84 kB raw at verification time).
- Runtime: pretrained, V1, V2, and V3 each produced a real mono 16 kHz WAV and a
  local Whisper transcript on CPU.
- Browser: a four-model locked fixture and a two-model custom-term comparison
  completed with playable audio, ASR, scores, metrics, provenance, and no console
  errors.
- Responsive: 375, 768, 1024, and 1440 CSS-pixel widths had no horizontal
  document overflow; the Experiment tab and comparison action remained visible.
- Accessibility: semantic regions/labels, SVG chart description plus exact table,
  44px targets, visible keyboard focus, `aria-live`, and reduced-motion rules were
  verified.
- Evidence screenshot: `docs/images/stage12-experiment.png`.
