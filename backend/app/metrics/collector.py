"""Latency, memory, RTF, and cold/warm metrics boundary."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Backend-internal metric values for one inference operation."""

    latency_ms: float
    real_time_factor: float
    memory_mb: float
    cold_start: bool


class MetricsCollector:
    """Collect process and inference metrics independently of HTTP handling."""

    def collect(self) -> MetricSnapshot:
        """Collect a snapshot once measurement instrumentation is implemented."""
        raise NotImplementedError("Metrics collection is not implemented yet.")

