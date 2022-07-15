import abc
import dataclasses
from typing import (
    Generic,
    Union,
    TypeVar,
    TYPE_CHECKING,
    Optional
)

TLocation = TypeVar('TLocation')

if TYPE_CHECKING:
    from slowmatch.compressed_edge import CompressedEdge
    from slowmatch.graph import DetectorNode
    from slowmatch.graph_fill_region import GraphFillRegion


@dataclasses.dataclass
class BlossomImplodeEvent:
    time: float
    blossom_region: 'GraphFillRegion'
    in_parent_region: 'GraphFillRegion'
    in_child_region: 'GraphFillRegion'


@dataclasses.dataclass
class RegionHitRegionEvent:
    time: float
    region1: 'GraphFillRegion'
    region2: 'GraphFillRegion'
    edge: 'CompressedEdge'

    def __post_init__(self):
        assert self.region1.id != self.region2.id
        if self.region1.id > self.region2.id:
            self.region1, self.region2 = self.region2, self.region1
            self.edge = self.edge.reversed()


@dataclasses.dataclass
class RegionHitBoundaryEvent:
    time: float
    region: 'GraphFillRegion'
    edge: 'CompressedEdge'


MwpmEvent = Union[BlossomImplodeEvent, RegionHitRegionEvent, RegionHitBoundaryEvent]


@dataclasses.dataclass
class TentativeEvent(Generic[TLocation]):
    time: float
    event_id: int

    @property
    @abc.abstractmethod
    def is_invalidated(self):
        pass

    @abc.abstractmethod
    def invalidate(self):
        pass

    def __lt__(self, other):
        if not isinstance(other, TentativeEvent):
            return NotImplemented
        return self.time < other.time


@dataclasses.dataclass
class TentativeNeighborInteractionEvent(TentativeEvent):
    location_data_1: 'DetectorNode'
    schedule_list_index_1: int
    location_data_2: Optional['DetectorNode']
    schedule_list_index_2: Optional[int]
    is_invalidated: bool = False

    def invalidate(self):
        self.is_invalidated = True
        assert self.location_data_1.schedule_list[self.schedule_list_index_1] is self
        self.location_data_1.schedule_list[self.schedule_list_index_1] = None
        if self.schedule_list_index_2 is not None:
            assert self.location_data_2.schedule_list[self.schedule_list_index_2] is self
            self.location_data_2.schedule_list[self.schedule_list_index_2] = None

    def __repr__(self):
        return f"TentativeNeighborInteraction({self.time},{self.event_id}," \
               f"DetectorNode({self.location_data_1.loc}),DetectorNode({self.location_data_2.loc})," \
               f"{self.is_invalidated})"


@dataclasses.dataclass
class TentativeRegionShrinkEvent(TentativeEvent):
    region: 'GraphFillRegion'
    is_invalidated: bool = False

    def invalidate(self):
        if not self.is_invalidated:
            assert self.region.shrink_event is self
            self.is_invalidated = True

