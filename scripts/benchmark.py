"""Benchmark sequential vs concurrent batch processing.

Run:
    python3 scripts/benchmark.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.concurrency.pipeline import process_batch


async def fake_io_task(item: int) -> int:
    """Simulate I/O-bound work such as API/network calls."""
    await asyncio.sleep(0.2)
    return item * item


async def run_sequential(items: list[int]) -> list[int]:
    results: list[int] = []
    for item in items:
        results.append(await fake_io_task(item))
    return results


async def run_concurrent(items: list[int]) -> list[int | BaseException]:
    return await process_batch(items, fake_io_task, concurrency=5)


async def main() -> None:
    items = list(range(20))

    start = time.perf_counter()
    sequential_results = await run_sequential(items)
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    concurrent_results = await run_concurrent(items)
    concurrent_time = time.perf_counter() - start

    print("Benchmark: sequential vs concurrent")
    print(f"Items: {len(items)}")
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Concurrent time: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
    print(f"Results equal: {sequential_results == concurrent_results}")


if __name__ == "__main__":
    asyncio.run(main())