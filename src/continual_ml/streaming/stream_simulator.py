"""Rate-controlled stream simulator — the Kafka stand-in.

Wraps any ``BaseDataSource`` and yields its records with realistic pacing. Two
modes:

* ``fixed``     — emit at a constant ``rate_per_sec``.
* ``time_warp`` — honor the gaps between record timestamps (so rush-hour bursts
  look like rush-hour bursts), compressed by ``time_warp_factor``.

Exposes live controls (pause/resume/set-rate) so the frontend can drive it. When
a real message broker is introduced later, it implements this same async
interface and nothing downstream changes.
"""

from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator, Optional

from continual_ml.config import StreamConfig
from continual_ml.data_sources.base_data_source import BaseDataSource
from continual_ml.schemas import Record


class StreamSimulator:
    def __init__(self, source: BaseDataSource, config: StreamConfig):
        self._source = source
        self._cfg = config
        self._rate = max(config.rate_per_sec, 0.1)
        self._running = config.autostart
        self._stopped = False
        self._emitted = 0

    # --- live controls (called by the API/frontend) --------------------------
    def pause(self) -> None:
        self._running = False

    def resume(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._stopped = True

    def set_rate(self, rate_per_sec: float) -> None:
        self._rate = max(float(rate_per_sec), 0.1)

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "stopped": self._stopped,
            "rate_per_sec": self._rate,
            "mode": self._cfg.mode,
            "emitted": self._emitted,
        }

    # --- pacing --------------------------------------------------------------
    def _fixed_delay(self) -> float:
        delay = 1.0 / self._rate
        if self._cfg.jitter > 0:
            delay *= 1.0 + random.uniform(-self._cfg.jitter, self._cfg.jitter)
        return max(delay, 0.0)

    def _warp_delay(self, prev: Optional[Record], cur: Record) -> float:
        if prev is None:
            return 0.0
        gap = (cur.timestamp - prev.timestamp).total_seconds()
        return max(min(gap / self._cfg.time_warp_factor, 5.0), 0.0)

    async def stream(self) -> AsyncIterator[Record]:
        """Yield records with pacing, respecting pause/stop/rate at runtime."""
        prev: Optional[Record] = None
        for record in self._source.stream():
            if self._stopped:
                break

            # Honor pause without busy-spinning.
            while not self._running and not self._stopped:
                await asyncio.sleep(0.1)
            if self._stopped:
                break

            if self._cfg.mode == "time_warp":
                delay = self._warp_delay(prev, record)
            else:
                delay = self._fixed_delay()
            if delay > 0:
                await asyncio.sleep(delay)

            prev = record
            self._emitted += 1
            yield record
