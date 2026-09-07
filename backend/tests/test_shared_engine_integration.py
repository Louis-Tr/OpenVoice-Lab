"""Real shared-session CPU inference and deterministic SpeechT5 concurrency."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from app.config.settings import Settings
from app.main import create_app


@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["kokoro-fp32", "kokoro-fp16", "speecht5-pretrained"])
def test_real_shared_engine_concurrent_audio(tmp_path, model_id):
    app = create_app(
        Settings(
            environment="test",
            generated_audio_dir=tmp_path / "audio",
            stage12_artifact_root=tmp_path / "stage12",
            product_cpu_threads=2,
        )
    )
    model = app.state.model_registry.get(model_id)
    if not model.artifacts_available:
        pytest.skip("Provision local serving artifacts for this model.")
    scheduler = app.state.engine_scheduler
    if scheduler.resources.cpu_units < 4:
        pytest.skip("Concurrent two-thread inference requires four usable CPU units.")
    if scheduler.resources.available_memory() < 2 * scheduler.resources.estimate_required_memory(
        model_id
    ):
        pytest.skip("Insufficient memory headroom for two concurrent reservations.")
    texts = ["Open Voice Lab speaks clearly.", "A different sentence for the second request."]
    with scheduler.acquire(model) as lease:
        shared = lease.loaded.value
        baseline = [shared.synthesize(text, model.voices[0]).samples for text in texts]
    barrier = Barrier(2)

    def run(text):
        with scheduler.acquire(model) as lease:
            assert lease.warm
            assert lease.loaded.value is shared
            barrier.wait(20)
            return lease.loaded.value.synthesize(text, model.voices[0]).samples

    with ThreadPoolExecutor(2) as pool:
        a, b = [pool.submit(run, text) for text in texts]
        results = [a.result(120), b.result(120)]
    for expected, actual in zip(baseline, results, strict=True):
        assert actual.ndim == 1 and actual.size > 0 and np.isfinite(actual).all()
        if model_id == "speecht5-pretrained":
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert app.state.model_loader.load_count(model_id) == 1
    assert scheduler.resources.reserved_cpu_units == 0
    assert scheduler.resources.reserved_memory_mb == 0
