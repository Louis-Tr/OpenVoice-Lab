# Resource-aware processing queue implementation plan

Status: core implementation completed on `codex/resource-aware-processing-queue`.
Baseline inspected: `3cfc458` (Audio8 removed from the active catalog).

Implemented in this branch: container-aware probing, conservative model
profiles, atomic reservations, FIFO backfilling with the 30-second strict
next-start rule, bounded queue payloads, queue expiry, client-scoped
idempotency, lease-safe model lifecycle, asynchronous synthesis APIs, aggregate
diagnostics, shared benchmark/experiment admission, graceful shutdown, and the
multi-request Angular workflow. Profile calibration and adaptive concurrency
remain deployment gates because they require isolated cold/warm measurements
from the target 4-vCPU/8-GiB Cloud Run instance; concurrency stays bounded by
the conservative static profiles until that evidence exists.

## 1. Objective and agreed scheduling policy

Increase completed synthesis throughput within the backend's CPU and memory
budget while preserving responsiveness, input validation, and request fairness.
Supported product models are Kokoro FP32, FP16, INT8, and SpeechT5.

The agreed policy is resource-aware backfilling with a **30-second aging
threshold and strict next-start reservation**:

1. Measure wait time from successful enqueue using a monotonic clock.
2. Before 30 seconds, scan pending requests in arrival order and dispatch the
   first request that fits. A later request may bypass an earlier non-fitting one.
3. At or after 30 seconds, the oldest eligible waiting request becomes reserved.
4. While a reservation exists, start no other compute work, even if it fits.
5. Let already-admitted work finish; start the reserved request when it fits.
6. After dispatch, clear the reservation and reevaluate all waiting requests.
7. If several requests are overdue, oldest enqueue time wins, with a sequence
   number as a deterministic tie-breaker.

The timer guarantees next-start priority, not a start within 30 seconds. A
request that cannot fit even on an otherwise idle instance is rejected instead
of becoming a permanent reservation. Cancellation or expiry clears reservation.

For this policy, "start" means atomic admission into model loading/execution.
Work already admitted before the threshold may complete its loading and inference.

## 2. Scope and architecture

First release: one scheduler per backend instance, one Uvicorn application
worker, CPU inference, bounded in-memory jobs, and no automatic model replication.
Each existing engine retains one inference slot. Different models may overlap
when their combined estimates fit. Same-model traffic remains serialized in
this release; increasing that concurrency requires separately measured replicas
or a demonstrably concurrent engine.

All backend-managed compute must acquire capacity from the same scheduler:
product synthesis, benchmark subprocesses, experiment inference, and experiment
ASR. Existing independently configured job services must not bypass accounting.

Proposed modules:

| Module | Responsibility |
| --- | --- |
| `backend/app/resources/probe.py` | Container/host resource limits and measurements |
| `backend/app/resources/profiles.py` | Model memory/CPU estimates and calibration records |
| `backend/app/resources/manager.py` | Reservation ledger and admission decisions |
| `backend/app/scheduling/policy.py` | Pure selection policy, aging, and reservation rules |
| `backend/app/scheduling/service.py` | Event loop, dispatch, completion, and shutdown |
| `backend/app/scheduling/store.py` | Bounded job state, snapshots, and expiry |
| `backend/app/schemas/jobs.py` | Public job/status contracts |
| `backend/app/api/synthesis_jobs.py` | Submit, inspect, and cancel jobs |

Existing integration points: `main.py`, `synthesis/service.py`, `models/loader.py`,
`inference/kokoro_onnx.py`, `inference/speecht5.py`, `benchmark/service.py`,
`benchmark/runner.py`, `experiments/jobs.py`, and `inference/speecht5_cpu.py`.

## 3. Resource discovery and measurement

Implement an injectable probe so tests do not depend on live machine load.

- Linux containers: resolve applicable cgroup paths and ancestors, supporting
  v2 and v1; account for memory limits, CPU quotas, and allowed CPU sets. Handle
  missing/unlimited values and nested limits explicitly.
- Local Windows/Linux: use configured budgets and available system capacity;
  reserve headroom for other applications. Never equate host RAM with memory
  allocated to a cloud container.
- Allow explicit CPU and memory budget overrides. Use the smaller of verified
  hard limits and configured limits. Report which source determined each limit.
- Sample approximately once per second using nonblocking CPU deltas. Normalize
  CPU usage to the effective quota, including fractional CPU quotas.
- Measure container memory where available. Account for children, cached
  engines, audio buffers, and memory-backed files. Use process-tree measurements
  only as a documented fallback; do not add them again to a container total.
- Track measurement freshness, CPU throttling where available, peak memory,
  and API responsiveness. Stale/invalid readings prevent new parallel admissions
  until a valid reading or conservative fallback is available.

Admission estimates are not an OS-level guarantee against OOM. Unexpected
allocations and native-runtime retention require headroom and pressure handling.

## 4. Model profiles and reservations

Profile each model variant separately. Record artifact/runtime fingerprint,
resident memory, peak load memory, additional inference memory by input-size
bucket, CPU allocation, and estimated duration. Include vocoders, ASR models,
subprocess interpreter overhead, encoding buffers, and output storage as needed.

Establish initial profiles with isolated cold and warm measurements on the
target deployment. Start conservatively; unknown profiles do not qualify for
optimistic parallel execution. Never infer peak memory from model file size.

For an unloaded engine, reserve for the larger of its peak loading requirement
and resident-plus-inference requirement. For a loaded engine, resident memory
is already accounted for; reserve only additional work. Concurrent waiters for
one model must share one loading operation, not allocate duplicate engines.

The reservation ledger tracks:

- Every resident engine and its active leases.
- Every admitted job's loading, inference, and encoding commitments.
- CPU allocations and available engine slots.
- Unmaterialized reserved memory growth versus already-observed allocations.

Admission requires measured memory plus outstanding, not-yet-materialized
growth plus the candidate's incremental peak to fit below the memory ceiling
minus headroom. Reconciliation must not double-count resident or active memory.
If attribution is uncertain, conservatively overreserve and record the reason.

Admission and reservation updates are atomic. Slow model loading, eviction,
inference, and file I/O happen outside the scheduler lock. Every real completion
or failure releases its lease exactly once. A cancelled coroutine is not proof
that its native inference has stopped.

Profile learning uses isolated samples for attribution and conservative observed
peaks. Do not attribute the whole process's memory increase to one job when
multiple jobs overlap. Raise estimates promptly after underestimation; lower
them only after sufficient evidence. Version profiles when runtimes change.

## 5. Model lifecycle and CPU controls

Refactor `ModelLoader` to expose engine states: unloaded, loading, idle, busy,
and unloading. Acquire/release leases around complete execution.

- Never evict an engine with active leases or a pending load.
- Evict eligible idle engines before constructing a replacement when needed.
- Prefer idle engines without queued demand, then least recently used engines.
- Observe actual memory after unloading; `gc.collect()` does not establish
  that a native allocator returned all memory to the operating system.
- Replace the one-model cache policy with a memory-budgeted policy; keep an
  optional engine-count ceiling as an additional constraint.
- The scheduler checks engine availability before dispatch, so same-engine
  waiters do not occupy worker threads blocked on an inference lock.

Pass CPU settings to Kokoro's ONNX sessions and configure SpeechT5's PyTorch
thread settings at process startup. Do not mutate process-global PyTorch thread
counts per overlapping request. Use a bounded executor owned by the scheduler.
CPU reservations control admission, not hard thread isolation; measure actual
contention and API latency to validate the estimates.

## 6. Dispatcher algorithm

Wake on enqueue, completion, cancellation, pressure changes, shutdown, and the
next aging deadline. Use a timer for the 30-second boundary; do not wait for
another request or inference completion to notice that the threshold passed.

```text
under scheduler lock:
    refresh measurements and reconcile completed resource changes
    remove expired/cancelled pending jobs
    reject jobs permanently impossible under the configured resource budget

    reserved = oldest pending job with wait >= 30 seconds, if any
    if reserved exists:
        if resources and engine slot fit reserved:
            reserve resources; mark admitted; dispatch reserved
            immediately reevaluate (another overdue job may now reserve)
        else:
            prepare safe idle-engine eviction if needed
            start no other compute job
        return

    while capacity exists:
        reevaluate overdue requests using the current clock
        if any request is now overdue:
            apply reservation branch above
        candidate = first fitting pending job in enqueue order
        if no candidate exists:
            consider a safe idle-engine eviction plan; stop dispatch loop
        atomically reserve and mark candidate admitted
        dispatch outside lock
```

Loading requirements and engine availability are part of "fits". Do not sort
the entire queue by smallest job or keep favoring warm models over fitting older
jobs. A finite queue permits a simple O(n) scan; no optimization solver needed.

## 7. Job lifecycle and HTTP contracts

Recommended public flow: asynchronous jobs, plus a compatible synchronous route.

| Endpoint | Behavior |
| --- | --- |
| `POST /api/synthesis/jobs` | Validate and enqueue; return 202 and a job ID |
| `GET /api/synthesis/jobs/{id}` | Return job status, timings, and final result |
| `POST /api/synthesis/jobs/{id}/cancel` | Cancel pending work or request cooperative cancellation |
| Existing `POST /api/synthesis` | Enqueue through the same scheduler and await its result |

Validate model, voice, input limits, and text preprocessing before enqueue where
possible. Exact tokenizer checks should use lightweight tokenizer validation,
not require loading heavyweight weights merely to reject an invalid request.
Capture immutable normalized text and selected settings in the queued job.

States: queued, reserved, loading, running, saving, completed, failed, cancelled.
For active jobs expose `cancellationRequested` until execution actually stops.
Expose queue wait, processing time, reservation state, and a machine-readable
waiting reason: CPU, memory, engine busy, or older reserved request.

Do not present queue rank as a guaranteed execution order or fabricate an ETA.
Completed responses retain the existing synthesis result/audio URL contract.
Support bounded, client-scoped idempotency keys to prevent duplicate submissions
after a network retry. Reject reuse with a different payload.

Queue overflow returns 429 with retry guidance. Invalid input returns 422;
unknown model remains 404. A valid model that cannot run within this deployment's
budget returns 503 with a resource-unavailable code. Accepted jobs that expire
or fail retain an inspectable terminal status for the configured retention time.

Queued jobs must not hold executor threads. HTTP disconnect does not cancel an
asynchronous job. For the synchronous compatibility route, cancel if still
queued on disconnect; if active, request cancellation and keep its resource
reservation until actual execution ends.

## 8. Frontend workflow

Update `synthesis-api.service.ts`, `synthesis.types.ts`, and the synthesis page
to submit asynchronous jobs and poll their status. Poll only active jobs, with
backoff on errors and cleanup on page destruction; SSE can follow if necessary.

- Keep model-specific input limits and the always-visible processed-text preview.
- Allow another valid submission after enqueue, rather than disabling the whole
  form until inference finishes. Preserve each job's own model/text/result.
- Show a compact list of the user's submitted jobs, with queued, reserved,
  running, failed, and completed states, and a cancel action.
- Select a completed job to view its audio and metrics. Out-of-order completion
  must not overwrite another job's result.
- Explain reservation as "Next to run; waiting for current work to finish."
- Surface overload/input errors with the user's draft intact.

Avoid exposing other users' texts in global queue diagnostics. Resource status
can report aggregate counts and utilization without request content.

## 9. Benchmarks and experiments

Experiments acquire scheduler leases for model loading/inference and ASR stages.
Shared vocoder/ASR caches count toward the same memory budget. Release active
stage leases at safe boundaries; preserve the experiment's result and cancellation
contracts. Every runnable stage receives its own enqueue timestamp. Already
running stages may finish after another request reserves next start.

Benchmark subprocesses currently run outside the serving process. Admit each
bounded per-model benchmark run through the parent scheduler, account for its
entire process footprint, and wait for process termination before releasing it.
Avoid creating a second independent scheduler inside an already-admitted child.

Preserve benchmark comparability by making a per-model benchmark run an
exclusive compute allocation in the first release. Its corpus size bounds that
allocation. An aged request may still wait for the active run to finish; the
30-second rule does not preempt it. Concurrent throughput load tests are a
separate mode and must not be mixed into isolated model-quality measurements.

Experiment recovery must resubmit unfinished work through the scheduler rather
than launching all recovered tasks immediately.

## 10. Failure, shutdown, and storage behavior

- Stop new admissions under sustained memory pressure; evict idle models where
  useful and reevaluate. Record pressure and estimate errors.
- Cancellation of native thread work is cooperative. Keep resources reserved
  until it actually returns. A future being cancelled cannot stop running work:
  [Python executor documentation](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.cancel).
- A response timeout must not release resources still consumed by inference.
  Hard execution termination would require isolated workers; defer that feature
  rather than claiming threads can be safely killed.
- On shutdown, stop acceptance, terminalize queued jobs, and drain admitted work
  within the hosting grace period. Do not promise completion across termination.
- Use bounded result retention and audio retention. Audio in `/tmp` consumes
  container resources and must be included in accounting. Do not delete an
  artifact while a retained completed job still promises access to it.
- Atomic WAV publication and content-based artifact identity should ensure a
  result points to its actual complete audio, including overlapping requests.

V1 is explicitly best-effort and instance-local: jobs and audio may disappear
when an instance is replaced. Redis/database coordination and shared object
storage are separate prerequisites for durable or multi-instance guarantees.

## 11. Initial configuration and tuning

Only the 30-second threshold is an agreed product requirement. Other values
below are proposed starting defaults and must be validated under load.

| Setting | Initial proposal |
| --- | --- |
| Aging threshold | 30 seconds |
| Pending queue capacity | 32 jobs, plus bounded total payload bytes |
| Resource sampling | 1 second, plus event-triggered scheduling |
| Queue wait timeout | 300 seconds, distinct from aging |
| Effective CPU/memory | Auto-detect with explicit override support |
| Memory safety headroom | Greater of 512 MiB or 20% of effective limit |
| Maximum active jobs | Start at 1; configurable hard ceiling of 4 |
| Per-engine inference slots | 1 |
| Initial CPU threads per engine | Calibrate; 2 is a starting point on 4 vCPU |
| Completed-job retention | Bounded count and TTL, coordinated with audio retention |

Start with deterministic admission and conservative profiles. Add adaptive
concurrency only after correctness tests and resource calibration pass:

1. Increase by one slot after a sustained underutilized interval with queued,
   otherwise eligible work and acceptable API latency.
2. On sustained pressure or throughput regression, decrease allowed concurrency
   multiplicatively, affecting only future admissions.
3. Add cooldown/hysteresis to avoid oscillation.
4. Never override memory checks, CPU reservations, engine slots, or aging priority.

Report effective utilization, not host-wide CPU percentages. High utilization
is not itself success if throughput decreases or tail latency becomes excessive.

## 12. Deployment integration

Keep one application worker and one Cloud Run instance for instance-local state.
Async processing needs CPU after a 202 response; verify instance-based CPU
allocation before enabling the browser job workflow. This does not provide job
durability or prevent instance termination:
[Cloud Run billing and CPU allocation](https://docs.cloud.google.com/run/docs/configuring/billing-settings).

HTTP concurrency and inference concurrency are different controls. Review the
existing Cloud Run concurrency of 1 so submissions/status/audio requests can
reach the app while work runs; choose the HTTP value through load testing, and
let the scheduler govern inference:
[Cloud Run concurrent requests](https://docs.cloud.google.com/run/docs/about-concurrency).

Expose the Git SHA, instance identifier, and effective scheduler configuration
in diagnostics so local/cloud differences are identifiable. Verify actual
deployment settings rather than assuming the Dockerfile overrides them.

## 13. Implementation milestones and acceptance gates

1. **Contracts and policy:** job states, configuration, fake clock/probe, pure
   backfilling policy. Gate: all fairness cases pass without real inference.
2. **Resource accounting:** limit detection, profiles, reservations, stale data
   handling. Gate: no double admission or double release under concurrent events.
3. **Model lifecycle:** leases, safe eviction/load ordering, CPU settings, bounded
   executor. Gate: no active eviction or duplicate same-model load.
4. **Synthesis integration:** both HTTP flows share the scheduler. Gate: valid
   jobs complete; overload and invalid input fail predictably; no thread waiters.
5. **Frontend jobs:** submission, polling, cancellation, result selection. Gate:
   switching models and out-of-order completions preserve the correct results.
6. **Other compute paths:** experiments, ASR, benchmark child processes, recovery.
   Gate: none can bypass the shared budget or the 30-second reservation rule.
7. **Load calibration and adaptation:** measure local and 4-vCPU/8-GB cloud
   profiles, tune thresholds, then enable bounded adaptive concurrency. Gate:
   measurable throughput benefit with no overload failures in the acceptance run.
8. **Rollout:** verify hosting configuration, deploy, inspect telemetry and
   revision identity, and retain a conservative serial-mode rollback switch.

Do not enable cloud parallel inference before milestones 1-6 are complete.

## 14. Verification matrix

Policy tests with a fake monotonic clock:

- A does not fit; B fits: B starts before A reaches 30 seconds.
- At exactly 30 seconds, A reserves next start and B cannot start even if it fits.
- A's threshold is noticed without any new requests or completions.
- Multiple overdue requests run in stable enqueue order.
- Cancelling or expiring the reserved request unblocks reevaluation.
- An impossible request never blocks the queue indefinitely.
- Already-admitted work completes without preemption or premature lease release.

Resource and lifecycle tests:

- Two admissions cannot claim the same free memory or CPU allocation.
- Fractional quotas, unlimited/missing cgroups, host fallback, and stale samples.
- Cold-load peaks, input expansion, encoding buffers, and retained native memory.
- Same-model requests load once and remain queued while its engine is busy.
- Different engines overlap only when both fit.
- Loader failure, inference failure, cancellation, and timeout release exactly once.
- Benchmark child exit and experiment ASR release their full reservations.

API/UI tests:

- Submission/status/cancel contracts, full queue, idempotency, bounded retention.
- Correct text/model/result association across multiple queued submissions.
- Existing synchronous synthesis and model-specific validation still work.
- Queued cancellation cannot start later; active cancellation reports honestly.
- Audio URL identifies the completed job's complete artifact.

Load scenarios: same-model bursts, mixed-model bursts, cold model switches,
maximum permitted input, sustained small jobs behind a large waiting job,
benchmark/experiment overlap, and an injected pressure/failure episode.

Measure completed jobs/minute, p50/p95 queue and end-to-end latency, CPU
utilization/throttling, peak memory, cold loads, evictions, overloads, estimate
errors, and fairness violations. Compare with a serial baseline on the same
hardware and workload. Require zero reservation-order violations and zero
resource-accounting leaks; set numerical throughput targets after baseline data.
