"""Golden input/output cases plus a runnable, text-only reviewer report.

From backend/: python tests/test_text_preprocessing.py --through 6
Each case declares the increment responsible for satisfying its contract.
No model is loaded and no network or application module is involved.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.text_processing.service import TextProcessingError, TextProcessingService

FIXTURES = Path(__file__).parent / "fixtures" / "text_preprocessing"


def load_cases():
    cases = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    for case in cases:
        for field in ("input", "expected"):
            if f"{field}File" in case:
                case[field] = (
                    (FIXTURES / case[f"{field}File"]).read_text(encoding="utf-8").rstrip("\n")
                )
    return cases


CASES = load_cases()


def evaluate(case):
    options = {
        "sanitize_text": case.get("sanitize", True),
        "normalize_text": case.get("normalize", True),
    }
    service = TextProcessingService()
    try:
        actual = service.process(case["input"], **options)
        return {
            "id": case["id"],
            "increment": case["increment"],
            "input": case["input"],
            "expected": case.get("expected"),
            "actual": actual,
            "passed": "error" not in case and actual == case["expected"],
            "idempotent": service.process(actual, **options) == actual,
        }
    except Exception as error:  # noqa: BLE001 - record unexpected crashes as failed review cases
        return {
            "id": case["id"],
            "increment": case["increment"],
            "input": case["input"],
            "expectedError": case.get("error"),
            "actualError": f"{type(error).__name__}: {error}",
            "passed": isinstance(error, TextProcessingError)
            and case.get("error", "__unexpected_error__") in str(error),
        }


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"inc{c['increment']}-{c['id']}")
def test_input_output(case):
    result = evaluate(case)
    assert result["passed"], json.dumps(result, ensure_ascii=False, indent=2)
    if "actual" in result:
        assert result["idempotent"], json.dumps(result, ensure_ascii=False, indent=2)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_both_disabled_preserves_exact_original(case):
    assert (
        TextProcessingService().process(case["input"], sanitize_text=False, normalize_text=False)
        == case["input"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, default=6)
    args = parser.parse_args()
    results = [evaluate(case) for case in CASES if case["increment"] <= args.through]
    report = {
        "through_increment": args.through,
        "total": len(results),
        "passed": sum(row["passed"] and row.get("idempotent", True) for row in results),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    sys.exit(0 if report["passed"] == report["total"] else 1)
