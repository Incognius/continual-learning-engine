"""Prequential (test-then-train) regression metrics.

Maintained incrementally so we never hold the full history. Tracks both
cumulative metrics (all samples) and rolling metrics (last ``window`` samples),
the latter being what actually reveals performance decay under drift.
"""

from __future__ import annotations

import math
from collections import deque


class RegressionMetrics:
    def __init__(self, window: int = 1000):
        self._n = 0
        self._sum_abs = 0.0
        self._sum_sq = 0.0
        # For cumulative R2 we track target mean/variance (Welford) + SSE.
        self._t_mean = 0.0
        self._t_m2 = 0.0
        self._sse = 0.0
        # Rolling buffers.
        self._abs = deque(maxlen=window)
        self._sq = deque(maxlen=window)
        self._y = deque(maxlen=window)

    def update(self, y_true: float, y_pred: float) -> None:
        err = y_true - y_pred
        abs_e, sq_e = abs(err), err * err

        self._n += 1
        self._sum_abs += abs_e
        self._sum_sq += sq_e
        self._sse += sq_e

        delta = y_true - self._t_mean
        self._t_mean += delta / self._n
        self._t_m2 += delta * (y_true - self._t_mean)

        self._abs.append(abs_e)
        self._sq.append(sq_e)
        self._y.append(y_true)

    @property
    def mae(self) -> float:
        return self._sum_abs / self._n if self._n else 0.0

    @property
    def rmse(self) -> float:
        return math.sqrt(self._sum_sq / self._n) if self._n else 0.0

    @property
    def r2(self) -> float:
        # 1 - SSE/SST, where SST is total variance of the target seen so far.
        if self._n < 2 or self._t_m2 <= 1e-12:
            return 0.0
        return 1.0 - (self._sse / self._t_m2)

    @property
    def rolling_mae(self) -> float:
        return sum(self._abs) / len(self._abs) if self._abs else 0.0

    @property
    def rolling_rmse(self) -> float:
        return math.sqrt(sum(self._sq) / len(self._sq)) if self._sq else 0.0

    @property
    def rolling_r2(self) -> float:
        # Honest "how is the model doing now" — immune to cold-start pollution.
        n = len(self._y)
        if n < 2:
            return 0.0
        mean_y = sum(self._y) / n
        sst = sum((y - mean_y) ** 2 for y in self._y)
        if sst <= 1e-12:
            return 0.0
        return 1.0 - (sum(self._sq) / sst)

    def snapshot(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "rolling_mae": self.rolling_mae,
            "rolling_rmse": self.rolling_rmse,
            "rolling_r2": self.rolling_r2,
            "n": float(self._n),
        }
