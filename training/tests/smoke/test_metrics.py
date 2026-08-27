from training.smoke.metrics import comparison_metrics, contains_phrase
from training.smoke.safety import (
    non_finite_training_values,
    require_successful_comparison,
    speech_labels_for_example,
    validate_training_safety,
)


def test_contains_phrase_is_word_order_sensitive() -> None:
    assert contains_phrase("Aspirin and chest pain.", "chest pain")
    assert not contains_phrase("pain in the chest", "chest pain")


def test_comparison_metrics_are_measured_without_claiming_missing_terms() -> None:
    metrics = comparison_metrics(
        [
            {
                "sample_id": "one",
                "reference": "take aspirin",
                "transcript": "take aspirin",
                "medical_terms": [{"canonical": "aspirin", "category": "medication"}],
                "inference_ms": 100.0,
                "rtf": 0.1,
                "peak_gpu_memory_mb": 500.0,
                "error": None,
            }
        ]
    )
    assert metrics["wer"] == 0.0
    assert metrics["domain_term_accuracy"]["accuracy"] == 1.0
    assert metrics["average_rtf"] == 0.1


def test_speecht5_labels_remove_exactly_one_batch_dimension() -> None:
    labels = [[[1.0, 2.0], [3.0, 4.0]]]
    assert speech_labels_for_example(labels) == labels[0]


def test_speecht5_labels_reject_ambiguous_batches() -> None:
    try:
        speech_labels_for_example([[[1.0]], [[2.0]]])
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("multi-example acoustic labels were accepted")


def test_training_safety_requires_supported_bf16_and_gradient_clipping() -> None:
    config = {
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 1.0,
        "checkpoint_steps": 5,
        "evaluation_steps": 5,
    }
    validate_training_safety(config, bf16_supported=True)
    try:
        validate_training_safety(config, bf16_supported=False)
    except RuntimeError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unsupported BF16 configuration was accepted")


def test_non_finite_training_values_are_detected() -> None:
    assert non_finite_training_values({"loss": 0.7, "grad_norm": 1.0}) == {}
    invalid = non_finite_training_values(
        {"loss": float("nan"), "grad_norm": float("inf")}
    )
    assert set(invalid) == {"loss", "grad_norm"}


def test_incomplete_comparison_cannot_report_success() -> None:
    summary = {
        "base": {"failure_count": 0},
        "adapted": {"failure_count": 1},
    }
    try:
        require_successful_comparison(summary)
    except RuntimeError as error:
        assert "incomplete" in str(error)
        assert "adapted" in str(error)
    else:
        raise AssertionError("comparison failures were accepted")

    require_successful_comparison(
        {"base": {"failure_count": 0}, "adapted": {"failure_count": 0}}
    )


def test_runpod_provisioning_covers_both_ssh_key_conventions() -> None:
    script = (
        __import__("pathlib").Path(__file__).parents[2]
        / "runpod"
        / "provision_smoke.ps1"
    ).read_text(encoding="utf-8")
    assert "SSH_PUBLIC_KEY" in script
    assert "PUBLIC_KEY" in script
    assert "@('COMMUNITY', 'SECURE')" in script
