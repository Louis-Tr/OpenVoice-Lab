"""Benchmark result aggregation boundary."""

from collections.abc import Sequence

from app.metrics.collector import MetricSnapshot


class BenchmarkEvaluator:
    """Aggregate per-sentence measurements into comparable results."""

    def aggregate(self, _measurements: Sequence[MetricSnapshot]) -> dict[str, float]:
        """Aggregate measurements after benchmark semantics are defined."""
        raise NotImplementedError("Benchmark evaluation is not implemented yet.")

