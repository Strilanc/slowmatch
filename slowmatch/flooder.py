import abc
import dataclasses
from typing import TypeVar, List, Generic, Optional, Tuple, Union, Iterable, Any

TLocation = TypeVar('TLocation')


class Flooder(Generic[TLocation], metaclass=abc.ABCMeta):
    """A system for reporting collisions and implosions of changing regions.

    Handles flood filling space from various points, and reporting when the
    flood fill regions interact. Must understand some basic rules for merging
    cycles of flooded regions into blossoms, and splitting them back apart when
    the blossom shrinks past its creation point.

    Contract:
        All regions must have a growth rate of +1, 0, or -1.
        When a region or blossom is created, its initial growth rate is +1.
        When a blossom is created, its underlying regions go to growth rate 0.
        Two touching regions with opposite growth rates produce no events.
        A blossom region implodes, and is removed, when its radius hits 0.
        Calling `next_event` advances time.
    """

    @abc.abstractmethod
    def create_region(self, location: TLocation) -> int:
        """Creates a new zero-radius region starting from the given location.

        Args:
            location: The center of the region.

        Returns:
            An identifier for the created region.
        """
        pass

    @abc.abstractmethod
    def set_region_growth(self, region_id: int, *, new_growth: int):
        """Changes the growth rate of a region.

        Args:
            region_id: The region whose growth rate is being changed.
            new_growth: -1, 0, or +1.
        """
        pass

    @abc.abstractmethod
    def create_blossom(self, contained_region_ids: List[int]) -> int:
        """Packs an odd number of regions into a blossom.

        The packed regions get their growth rate set to 0, and will not be
        involved in any events. (Anything that would collide with them will hit
        the blossom instead.)

        Args:
            contained_region_ids: The regions that the blossom subsumes.

        Returns:
            An identifier for the created blossom region.
        """
        pass

    @abc.abstractmethod
    def next_event(self, max_time: float = float('inf')) -> Optional['MwpmEvent']:
        """Advances time until `max_time` or the next collision/implosion.

        Args:
            max_time: If set, the progression of time will not go beyond this
                point. If there are no events by `max_time`, then `None` is
                returned. If `max_time` is not set, it defaults to positive
                infinity.

        Returns:
            The event that stopped the progression of time. If there are no more
            events to be had, or `max_time` is hit before an interesting event,
            then the result is None.

            When events occur at the same time, one is returned arbitrarily.
        """
        pass


@dataclasses.dataclass
class BlossomImplodeEvent:
    time: float
    blossom_region_id: int
    in_out_touch_pairs: Iterable[Tuple[int, int]]

    def __post_init__(self):
        self.in_out_touch_pairs = tuple(sorted(self.in_out_touch_pairs))


@dataclasses.dataclass
class RegionHitRegionEvent:
    time: float
    region1: int
    region2: int

    def __post_init__(self):
        assert self.region1 != self.region2
        if self.region1 > self.region2:
            self.region1, self.region2 = self.region2, self.region1


@dataclasses.dataclass
class RegionHitBoundaryEvent:
    time: float
    region: int
    boundary: Any


MwpmEvent = Union[BlossomImplodeEvent, RegionHitRegionEvent, RegionHitBoundaryEvent]
