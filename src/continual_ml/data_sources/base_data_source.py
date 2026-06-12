"""The data-source abstraction — the single seam for plugging in new datasets.

Any dataset (synthetic, NYC taxi, electricity, crypto, intrusion) becomes usable
by implementing this interface. Nothing downstream of the source knows or cares
which concrete dataset is flowing through it; they read ``Record``s and a
``SourceSchema``. That is what makes "swap the dataset" a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from continual_ml.schemas import Record, SourceSchema


class BaseDataSource(ABC):
    """Abstract base class for every data source.

    Contract:
      * ``schema()`` describes the feature/target layout once, up front.
      * ``stream()`` yields ``Record``s in the order they should be processed
        (for time series, that means chronological order).

    Implementations should be *lazy* — ``stream()`` returns an iterator and must
    not load an entire dataset into memory if it can be avoided, because the
    consumer (the engine) pulls one record at a time.
    """

    @abstractmethod
    def schema(self) -> SourceSchema:
        """Return the source's feature/target description."""
        raise NotImplementedError

    @abstractmethod
    def stream(self) -> Iterator[Record]:
        """Yield records one at a time, in processing order."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.schema().name

    def __iter__(self) -> Iterator[Record]:
        return self.stream()
