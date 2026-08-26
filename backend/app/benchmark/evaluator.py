"""Benchmark result aggregation boundary."""

from collections.abc import Sequence
from math import ceil, floor
from statistics import mean, median

from app.schemas.benchmark import BenchmarkAggregate, BenchmarkCaseResult
from app.schemas.model import ModelSummary


class BenchmarkEvaluator:
    """Aggregate successful measurements while retaining failure counts."""

    def aggregate(
        self,
        raw_results: Sequence[BenchmarkCaseResult],
        models: Sequence[ModelSummary],
    ) -> list[BenchmarkAggregate]:
        """Return one comparable aggregate for every requested model."""
        aggregates: list[BenchmarkAggregate] = []
        for model in models:
            model_results = [result for result in raw_results if result.model_id == model.id]
            successful = [
                result for result in model_results if result.status == "success" and result.metrics
            ]
            latencies = [result.metrics.inference_ms for result in successful if result.metrics]
            real_time_factors = [
                result.metrics.real_time_factor for result in successful if result.metrics
            ]
            memory_readings = [
                result.metrics.memory_mb for result in successful if result.metrics
            ]

            aggregates.append(
                BenchmarkAggregate(
                    model_id=model.id,
                    name=model.name,
                    precision=model.precision,
                    model_variant=model.variant,
                    total_cases=len(model_results),
                    success_count=len(successful),
                    failure_count=sum(
                        result.status == "failure" for result in model_results
                    ),
                    average_latency_ms=self._average(latencies, digits=3),
                    median_latency_ms=self._median(latencies, digits=3),
                    p95_latency_ms=self._percentile(latencies, 0.95, digits=3),
                    average_real_time_factor=self._average(
                        real_time_factors,
                        digits=6,
                    ),
                    average_memory_mb=self._average(memory_readings, digits=3),
                    peak_memory_mb=(
                        round(max(memory_readings), 3) if memory_readings else None
                    ),
                )
            )
        return aggregates

    @staticmethod
    def _average(values: Sequence[float], *, digits: int) -> float | None:
        return round(mean(values), digits) if values else None

    @staticmethod
    def _median(values: Sequence[float], *, digits: int) -> float | None:
        return round(median(values), digits) if values else None

    @staticmethod
    def _percentile(
        values: Sequence[float],
        percentile: float,
        *,
        digits: int,
    ) -> float | None:
        """Calculate a deterministic linearly interpolated percentile."""
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower = floor(position)
        upper = ceil(position)
        if lower == upper:
            return round(ordered[lower], digits)
        weight = position - lower
        interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * weight
        return round(interpolated, digits)
