"""Concurrency tests for BatchPipeline."""

from __future__ import annotations

import asyncio

import pytest

from src.concurrency.pipeline import BatchPipeline, process_batch


class TestBatchPipeline:

    @pytest.mark.asyncio
    async def test_returns_results_in_order(self):
        async def task(n: int) -> int:
            await asyncio.sleep(0)
            return n * 2

        pipeline = BatchPipeline(concurrency=4)
        results = await pipeline.run([task(i) for i in range(5)])
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_isolates_failures(self):
        async def good() -> str:
            return "ok"

        async def bad() -> str:
            raise ValueError("task failed")

        pipeline = BatchPipeline(concurrency=4)
        results = await pipeline.run([good(), bad(), good()], return_exceptions=True)
        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ok"

    @pytest.mark.asyncio
    async def test_bounded_concurrency(self):
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def track() -> None:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1

        pipeline = BatchPipeline(concurrency=3)
        await pipeline.run([track() for _ in range(9)])
        assert max_active <= 3

    @pytest.mark.asyncio
    async def test_empty_task_list(self):
        pipeline = BatchPipeline(concurrency=2)
        results = await pipeline.run([])
        assert results == []

    @pytest.mark.asyncio
    async def test_invalid_concurrency_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BatchPipeline(concurrency=0)

    @pytest.mark.asyncio
    async def test_process_batch_convenience(self):
        async def double(x: int) -> int:
            return x * 2

        results = await process_batch([1, 2, 3], double, concurrency=2)
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_all_fail_returns_exceptions(self):
        async def always_fail() -> None:
            raise RuntimeError("boom")

        pipeline = BatchPipeline(concurrency=2)
        results = await pipeline.run([always_fail() for _ in range(3)], return_exceptions=True)
        assert all(isinstance(r, RuntimeError) for r in results)
