import abc
import dataclasses
from typing import TypeVar, List, Generic, Optional, Tuple

import cirq

TLocation = TypeVar('TLocation')


class Event(metaclass=abc.ABCMeta):
    def __init__(self, *, time: float):
        self.time = time


@cirq.value_equality
class BlossomImplodeEvent(Event):
    def __init__(self,
                 *,
                 time: float,
                 blossom_region: int,
                 external_touching_region_1: int,
                 internal_touching_region_1: int,
                 external_touching_region_2: int,
                 internal_touching_region_2: int):
        super().__init__(time=time)
        self.external_touching_region_1 = external_touching_region_1
        self.internal_touching_region_1 = internal_touching_region_1
        self.external_touching_region_2 = external_touching_region_2
        self.internal_touching_region_2 = internal_touching_region_2
        self.blossom_region = blossom_region

    def _value_equality_values_(self):
        return (
            self.time,
            self.external_touching_region_1,
            self.external_touching_region_2,
            self.internal_touching_region_1,
            self.internal_touching_region_2,
        )


@cirq.value_equality
class RegionHitRegionEvent(Event):
    def __init__(self, *, time: float, region1: int, region2: int):
        super().__init__(time=time)
        self.region1 = region1
        self.region2 = region2

    def _value_equality_values_(self):
        return self.time, self.region1, self.region2


class FillSystem(Generic[TLocation], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_region(self, location: TLocation) -> int:
        pass

    @abc.abstractmethod
    def set_region_growth(self, region_id: int, *, new_growth: int):
        pass

    @abc.abstractmethod
    def create_blossom(self, contained_region_ids: List[int]) -> int:
        pass

    @abc.abstractmethod
    def next_event(self, max_time: Optional[float] = None) -> Event:
        pass


@dataclasses.dataclass
class VaryingRadius:
    """A region radius that is varying linearly over time."""

    time0: float
    growth: int = dataclasses.field(default=1)
    radius0: float = dataclasses.field(default=0)

    def __add__(self, other: 'VaryingRadius') -> 'VaryingRadius':
        result = VaryingRadius(time0=self.time0, growth=self.growth, radius0=self.radius0)
        result.sync(other.time0)
        result.growth += other.growth
        result.radius0 += other.radius0
        return result

    def radius_at(self, time: float):
        return (time - self.time0) * self.growth + self.radius0

    def sync(self, new_time: float, new_growth: Optional[int] = None):
        self.radius0 = self.radius_at(new_time)
        self.time0 = new_time
        if new_growth is not None:
            self.growth = new_growth
