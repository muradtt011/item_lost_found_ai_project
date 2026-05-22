
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BatchPipeline:
    """Run async tasks with bounded concurrency."""

    def __init__(self, concurrency: int = 4) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run(
        self,
        tasks: list[Coroutine[Any, Any, T]],
        *,
        return_exceptions: bool = True,
    ) -> list[T | BaseException]:
        """Execute tasks concurrently, returning results in input order."""
        logger.debug(
            "BatchPipeline.run: %d tasks, concurrency=%d",
            len(tasks),
            self._concurrency,
        )

        async def _guarded(idx: int, coro: Coroutine[Any, Any, T]) -> tuple[int, T | BaseException]:
            async with self._semaphore:
                try:
                    result = await coro
                    logger.debug("Task %d completed OK", idx)
                    return idx, result
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Task %d failed: %s", idx, exc)
                    if return_exceptions:
                        return idx, exc
                    raise

        wrapped = [_guarded(i, task) for i, task in enumerate(tasks)]

        if return_exceptions:
            pairs = await asyncio.gather(*wrapped)
        else:
            pairs = cast(
                list[tuple[int, T | BaseException]],
                await asyncio.gather(*wrapped, return_exceptions=False),
            )

        ordered = sorted(pairs, key=lambda pair: pair[0])
        results = [value for _, value in ordered]

        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            logger.warning(
                "BatchPipeline finished: %d/%d tasks failed",
                len(failures),
                len(tasks),
            )
        else:
            logger.info("BatchPipeline finished: all %d tasks OK", len(tasks))

        return results


async def process_batch(
    items: list[Any],
    processor: Callable[[Any], Coroutine[Any, Any, T]],
    concurrency: int = 4,
) -> list[T | BaseException]:
    """Apply an async processor to each item concurrently."""
    pipeline = BatchPipeline(concurrency=concurrency)
    tasks = [processor(item) for item in items]
    return await pipeline.run(tasks)