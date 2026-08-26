"""Benchmark execution orchestration boundary."""

from app.schemas.benchmark import BenchmarkRequest, BenchmarkResult


class BenchmarkRunner:
    """Execute the predefined corpus through the synthesis workflow."""

    async def run(self, _request: BenchmarkRequest) -> BenchmarkResult:
        """Run a benchmark after execution policy is implemented."""
        raise NotImplementedError("Benchmark execution is not implemented yet.")

