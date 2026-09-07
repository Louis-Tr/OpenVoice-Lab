# Engine scheduling and resource admission

Main synthesis runs through one process-local `EngineScheduler`. It caches one
engine per model ID, counts active leases, and allows concurrent requests to share
the engine's read-only weights. Request tensors, generated audio and SpeechT5
random generators are independent. The engine remains protected through WAV
storage; cancelling an awaiting HTTP task does not release a running worker's lease.

`ModelLoader` validates artifacts and constructs engines. It no longer caches or
evicts them. The scheduler reserves resources before cold construction, includes
pending loads in its cache-slot count, and evicts only idle LRU entries before
replacement construction. Construction, native cleanup and inference execute
outside its admission lock. Concurrent requests for a model still loading receive
503; warm concurrent calls share the completed engine.

## Memory and CPU budgets

Each request, cold or warm, reserves its full static allowance:

| Model | Reservation (MiB) |
| --- | ---: |
| Kokoro FP32 | 1700 |
| Kokoro FP16 | 1500 |
| SpeechT5 pretrained | 1600 |
| Kokoro INT8 | Disabled pending maximum-input measurement |

These are conservative estimates based on observed **process RSS**, including
runtime and allocator memory. They are neither exclusively model-owned bytes nor
enforced allocation limits. Even warm requests reserve the full amount. Fixed
profiles do not change in response to runtime logs. Existing benchmark history
remains readable; INT8 is excluded from new default benchmark workloads.

Available memory is the smallest of host available RAM, applicable Linux cgroup
v1/v2 headroom (including accessible ancestors), and an optional configured
process ceiling minus current RSS. The scheduler subtracts the safety margin and
all outstanding request reservations. This intentionally double-counts memory
already reflected in current usage. If eviction does not reduce actual usage,
the next admission check still rejects the request.

| Environment variable | Default / meaning |
| --- | --- |
| `OPENVOICE_PRODUCT_CPU_UNITS` | Usable logical CPUs, bounded by affinity and container CPU quota; explicit values must not exceed that capacity. |
| `OPENVOICE_PRODUCT_CPU_THREADS` | `min(2, CPU_UNITS)`; CPU units reserved per request and runtime intra-op thread limit. |
| `OPENVOICE_PRODUCT_MEMORY_LIMIT_MB` | Unset; optional process RSS ceiling in MiB. |
| `OPENVOICE_PRODUCT_MEMORY_HEADROOM_MB` | `256` MiB withheld from admission. |
| `OPENVOICE_PRODUCT_MAXIMUM_CACHED_MODELS` | Existing cache-entry limit, including pending/retiring entries. Unset means no count limit; memory still gates admission. |

Fractional CPU quotas below one CPU allow one runnable thread, subject to OS
throttling. Units are scheduling tokens, not assigned cores or a hard limit on
every helper thread in the process. ONNX sessions use the CPU provider with fixed
intra-op threads, sequential graph execution and idle spinning disabled. Shared
sessions share their thread pools; per-request reservations remain conservative.

PyTorch is configured once before model work (intra-op=request allowance,
inter-op=1). Its thread pools are process-wide. Experiment runtime configuration
uses the same guard; an explicit conflicting thread setting fails clearly rather
than changing pools during active product inference. Use matching product and
experiment thread counts when both run in one process. Unspecified experiment
thread counts inherit the configured process setting.

Kokoro phonemization uses a process-wide lock because eSpeak has global state;
the expensive inference call runs outside that lock. SpeechT5 tokenization is
briefly locked, while generation runs concurrently. Its instance-bound decoder
prenet dropout adapter uses a request-local CPU generator seeded to 42. The
adapter requires the pinned serving Transformers 4.57.6 implementation. It does
not copy weights, reset global RNG state per request, or patch global functions.

## Errors, logs and operational limits

Admission failures return HTTP 503 with the existing `{"detail": "..."}` shape.
There is no request queue or admission timeout. Failures include insufficient
memory/CPU, all cache slots active, a pending load and missing memory profiles.
Inference failures keep the loaded engine cached and release their reservations.

Structured `engine_admitted`, `engine_rejected`, `engine_evicted` and
`engine_released` events report model ID, active users, reservations and the
model's memory estimate. Existing `tts_process_rss_measurement` events continue
to measure construction/cache lookup and inference boundaries. Admission and
eviction are excluded from model load timing. Overlapping inference samples
include allocations from other requests; they cannot attribute memory exclusively
to an individual request. Public synthesis success fields remain unchanged.

This budget coordinates main synthesis within **one process**. Benchmark workers
have their own scheduler and budget; live experiments retain their existing
lifecycle. Other workers/processes can consume resources after an admission
check. Use a single serving worker for a single shared cache and do not interpret
this scheduler as a host-wide resource governor. The existing deployment cache
limit of one allows parallel requests for that model but prevents another model
from loading while it is active.

Before deployment, repeat maximum-permitted-input tests for each enabled model
at the chosen thread count, then with concurrent same-model and mixed-model
requests. Record peak process RSS, valid audio, latency, CPU reservation counts
and any rejections. Confirm the reservations return to zero on completion and
failure. Allow headroom for transient peaks, which the 75 ms sampler can miss.
The previously supplied input strings are not retained as benchmark fixtures, so
new measurements must record their exact text and resulting audio duration.
