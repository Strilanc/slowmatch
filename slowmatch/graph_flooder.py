import dataclasses
import heapq
from typing import (
    Callable,
    TypeVar,
    Set,
    List,
    Tuple,
    Generic,
    Optional,
    Dict,
    Union,
    Iterable,
    Iterator,
    Any,
)

import cirq

from slowmatch.flooder import (
    MwpmEvent,
    Flooder,
    RegionHitRegionEvent,
    BlossomImplodeEvent,
    RegionHitBoundaryEvent,
)
from slowmatch.varying import Varying

TLocation = TypeVar('TLocation')


@dataclasses.dataclass
class TentativeEvent(Generic[TLocation]):
    kind: str
    time: int
    event_id: int
    involved_object: Tuple[Union['LocationData', 'GraphFillRegion'], ...]
    is_invalidated: bool = dataclasses.field(default=False)

    def invalidate(self):
        if not self.is_invalidated:
            self.is_invalidated = True
            for obj in self.involved_object:
                del obj.schedule_map[self.event_id]

    def __lt__(self, other):
        if not isinstance(other, TentativeEvent):
            return NotImplemented
        return (self.time, self.kind) < (other.time, other.kind)


@cirq.value_equality(unhashable=True)
class GraphFillRegion(Generic[TLocation]):
    def __init__(
        self,
        *,
        id: int,
        source: Optional[TLocation] = None,
        radius: Union[int, float, Varying] = Varying.T,
        blossom_children: Optional[Iterable['GraphFillRegion']] = None,
    ):
        self.id = id
        self.source = source
        self.area: Set[TLocation] = set()
        self.radius = Varying(radius)
        self.blossom_children: Optional[List['GraphFillRegion', ...]] = (
            None if blossom_children is None else list(blossom_children)
        )
        self.schedule_map: Dict[int, TentativeEvent] = {}

    def arbitrary_rep(self) -> TLocation:
        if self.source is not None:
            return self.source
        return self.blossom_children[0].arbitrary_rep()

    def iter_all_sources(self) -> Iterator[TLocation]:
        if self.source is not None:
            yield self.source
        if self.blossom_children is not None:
            for child in self.blossom_children:
                yield from child.iter_all_sources()

    def _value_equality_values_(self):
        return self.id, self.source, self.radius, self.blossom_children

    def __repr__(self):
        return (
            f'GraphFillRegion('
            f'id={self.id!r}, '
            f'source={self.source!r}, '
            f'radius={self.radius!r}, '
            f'blossom_children={self.blossom_children!r})'
        )


@dataclasses.dataclass
class MileMarker:
    region: GraphFillRegion
    distance_from_region_center: float


@dataclasses.dataclass()
class LocationData(Generic[TLocation]):
    loc: TLocation
    mile_markers: List[MileMarker] = dataclasses.field(default_factory=list)
    schedule_map: Dict[int, TentativeEvent] = dataclasses.field(default_factory=dict)

    def has_same_owner_as(self, other: 'LocationData'):
        if not self.mile_markers or not other.mile_markers:
            return False
        return self.mile_markers[-1].region is other.mile_markers[-1].region

    def local_radius(self) -> Varying:
        if self.mile_markers:
            marker = self.mile_markers[-1]
            return marker.region.radius - marker.distance_from_region_center
        return Varying(0)


class GraphFlooder(Generic[TLocation], Flooder[TLocation]):
    def __init__(
        self,
        neighbors: Callable[[TLocation], List[Tuple[float, TLocation]]],
        boundary: Callable[[TLocation], bool] = lambda _: False,
    ):
        self._location_data_map: Dict[TLocation, LocationData] = {}
        self._region_data_map: Dict[int, GraphFillRegion] = {}
        self._next_region_id = 0
        self._next_event_id = 0
        self.time = 0
        self.neighbors = neighbors
        self.boundary = boundary
        self._sorted_schedule: List[TentativeEvent] = []

    def create_region(self, location: TLocation) -> int:
        k = self._next_region_id
        self._next_region_id += 1
        self._region_data_map[k] = GraphFillRegion(id=k, source=location)
        self._do_region_arriving_at_empty_location(
            region_id=k, location=location, distance_from_region_center=0,
        )
        return k

    def next_event(self, max_time: float = float('inf')) -> Optional[MwpmEvent]:
        # Process events until interaction is needed or no more work can be done.
        while self._sorted_schedule and self._sorted_schedule[0].time <= max_time:
            # Get next valid event.
            tentative_event: TentativeEvent = heapq.heappop(self._sorted_schedule)
            if tentative_event.is_invalidated:
                continue

            # Apply transformations based on the event.
            self.time = tentative_event.time
            if tentative_event.kind == '0_leave_location':
                mwpm_event = None
                self._do_region_leaving_location(location_data=tentative_event.involved_object[0],)
            elif tentative_event.kind == '1_neighbor_interaction':
                mwpm_event = self._do_neighbor_interaction(
                    loc_data_1=tentative_event.involved_object[0],
                    loc_data_2=tentative_event.involved_object[1],
                )
            elif tentative_event.kind == '2_implosion':
                mwpm_event = self._do_implosion(region=tentative_event.involved_object[0],)
            else:
                raise NotImplementedError(f'Unrecognized event kind: {tentative_event.kind}')

            # If the event requires an update to the MWPM state, return it.
            if mwpm_event is not None:
                return mwpm_event

        # Nothing more to do right now.
        return None

    def set_region_growth(self, region_id: int, *, new_growth: int):
        region = self._region_data_map[region_id]
        region.radius = region.radius.then_slope_at(time_of_change=self.time, new_slope=new_growth)
        self._reschedule_events_for_region(region)

    def create_blossom(self, contained_region_ids: List[int]) -> int:
        k = self._next_region_id
        self._next_region_id += 1
        blossom_region = GraphFillRegion(id=k)
        blossom_region.radius = Varying(base_time=self.time, slope=1)
        blossom_region.blossom_children = []
        self._region_data_map[k] = blossom_region

        for i in contained_region_ids:
            region = self._region_data_map[i]
            del self._region_data_map[i]
            blossom_region.blossom_children.append(region)
            region.radius = region.radius.then_slope_at(time_of_change=self.time, new_slope=0)
            for loc in region.area:
                blossom_region.area.add(loc)
                loc_data = self._loc_data(loc)
                loc_data.mile_markers.append(
                    MileMarker(blossom_region, -loc_data.local_radius()(self.time),)
                )

        self._reschedule_events_for_region(blossom_region)

        # Rescheduling the blossom region fixed location schedules, but not
        # child region schedules. Fix them now.
        for child in blossom_region.blossom_children:
            for ev in list(child.schedule_map.values()):
                ev.invalidate()

        return k

    def _schedule_tentative_event(
        self, *objects: Union['LocationData', 'GraphFillRegion'], kind: str, time: int
    ):
        tentative_event = TentativeEvent(
            kind=kind, time=time, event_id=self._next_event_id, involved_object=objects,
        )
        self._next_event_id += 1
        for obj in objects:
            obj.schedule_map[tentative_event.event_id] = tentative_event
        heapq.heappush(self._sorted_schedule, tentative_event)

    def _reschedule_events_for_region(self, region: 'GraphFillRegion'):
        for event in list(region.schedule_map.values()):
            event.invalidate()

        if region.radius.slope < 0:
            self._schedule_tentative_event(
                region, time=region.radius.zero_intercept(), kind='2_implosion',
            )

        for loc in region.area:
            self._reschedule_events_at_location(location_data=self._loc_data(loc))

    def _reschedule_events_at_location(self, *, location_data: 'LocationData'):
        for event in list(location_data.schedule_map.values()):
            event.invalidate()
        assert not location_data.schedule_map

        rad1 = location_data.local_radius()
        if (
            rad1.slope < 0
            and location_data.mile_markers[-1].region.source != location_data.loc
            and len(location_data.mile_markers) == 1
        ):
            self._schedule_tentative_event(
                location_data,
                # Offset to leave before anything else.
                time=rad1.zero_intercept(),
                kind='0_leave_location',
            )

        for distance, neighbor in self.neighbors(location_data.loc):
            neighbor_location_data = self._loc_data(neighbor)
            if location_data.has_same_owner_as(neighbor_location_data):
                continue
            rad2 = neighbor_location_data.local_radius()
            rad3 = rad1 + rad2 - distance
            if rad3.slope <= 0:
                continue
            self._schedule_tentative_event(
                location_data,
                neighbor_location_data,
                time=rad3.zero_intercept(),
                kind='1_neighbor_interaction',
            )

    def _do_region_arriving_at_empty_location(
        self, *, region_id: int, location: TLocation, distance_from_region_center: float
    ):
        location_data = self._loc_data(location)
        region_data = self._region_data_map[region_id]
        assert region_data.radius.slope > 0

        assert not location_data.mile_markers
        location_data.mile_markers.append(
            MileMarker(region=region_data, distance_from_region_center=distance_from_region_center)
        )
        region_data.area.add(location)

        self._reschedule_events_at_location(location_data=location_data)

    def _do_region_leaving_location(self, *, location_data: 'LocationData'):
        region_data = location_data.mile_markers[-1].region
        assert region_data.radius.slope < 0
        assert len(location_data.mile_markers)
        assert location_data.mile_markers[0].region is region_data

        location_data.mile_markers.clear()
        region_data.area.remove(location_data.loc)

        self._reschedule_events_at_location(location_data=location_data)

    def _edge_distance(self, loc_data_1: 'LocationData', loc_data_2: 'LocationData'):
        for distance, neighbor in self.neighbors(loc_data_1.loc):
            if neighbor == loc_data_2.loc:
                assert isinstance(distance, int)
                return distance
        # coverage: ignore
        raise ValueError(f'Not adjacent: {loc_data_1.loc} and {loc_data_2.loc}')

    def _do_neighbor_interaction(
        self, *, loc_data_1: 'LocationData', loc_data_2: 'LocationData'
    ) -> Optional[MwpmEvent]:
        assert loc_data_1.mile_markers or loc_data_2.mile_markers
        assert not loc_data_1.has_same_owner_as(loc_data_2)

        # One region spreading into an empty location?
        if loc_data_2.mile_markers and not loc_data_1.mile_markers:
            loc_data_1, loc_data_2 = loc_data_2, loc_data_1
        if loc_data_1.mile_markers and not loc_data_2.mile_markers:
            if self.boundary(loc_data_2.loc):
                return RegionHitBoundaryEvent(
                    time=self.time,
                    region=loc_data_1.mile_markers[-1].region.id,
                    boundary=loc_data_2.loc,
                )
            else:
                d = loc_data_1.mile_markers[-1].distance_from_region_center + self._edge_distance(
                    loc_data_1, loc_data_2
                )
                self._do_region_arriving_at_empty_location(
                    region_id=loc_data_1.mile_markers[-1].region.id,
                    location=loc_data_2.loc,
                    distance_from_region_center=d,
                )
                return None

        # Two regions colliding.
        region1 = loc_data_1.mile_markers[-1].region
        region2 = loc_data_2.mile_markers[-1].region
        assert region1 is not region2
        assert region1.id != region2.id
        return RegionHitRegionEvent(region1=region1.id, region2=region2.id, time=self.time)

    def _loc_data(self, loc: TLocation):
        assert not isinstance(loc, LocationData)
        if loc not in self._location_data_map:
            self._location_data_map[loc] = LocationData(loc=loc)
        return self._location_data_map[loc]

    def _get_in_out_match_pairs(
        self, region: 'GraphFillRegion', region_marker_index: int
    ) -> List[Tuple[int, int]]:
        result = []
        for loc in region.area:
            loc_data = self._loc_data(loc)
            r1 = loc_data.local_radius()
            inner = loc_data.mile_markers[region_marker_index].region.id
            for distance, neighbor in self.neighbors(loc):
                neighbor_data = self._loc_data(neighbor)
                if not neighbor_data.mile_markers or loc_data.has_same_owner_as(neighbor_data):
                    continue
                r2 = r1 + neighbor_data.local_radius()
                if r2.slope < 0 or abs(r2(self.time) - distance) > 1e-8:
                    continue
                outer = neighbor_data.mile_markers[-1].region.id
                result.append((inner, outer))
        return result

    def _do_implosion(self, region: 'GraphFillRegion') -> Optional[MwpmEvent]:
        assert region.radius(self.time) == 0
        assert region.radius.slope < 0

        degenerate_collision = not region.blossom_children

        in_out = self._get_in_out_match_pairs(
            region, region_marker_index=-1 if degenerate_collision else -2
        )
        assert len(in_out) >= 2

        if degenerate_collision:
            return RegionHitRegionEvent(time=self.time, region1=in_out[0][1], region2=in_out[1][1])

        self._explode_region(region)
        return BlossomImplodeEvent(
            time=self.time, blossom_region_id=region.id, in_out_touch_pairs=in_out
        )

    def _explode_region(self, region: 'GraphFillRegion'):
        for loc in region.area:
            loc_data = self._loc_data(loc)
            assert len(loc_data.mile_markers) >= 2
            loc_data.mile_markers.pop()

        del self._region_data_map[region.id]
        for child in region.blossom_children:
            assert child.id not in self._region_data_map
            self._region_data_map[child.id] = child

    def _find_potentially_inactive_region(self, region_id: int) -> 'GraphFillRegion':
        if region_id in self._region_data_map:
            return self._region_data_map[region_id]

        queue = [*self._region_data_map.values()]
        while queue:
            e = queue.pop()
            if e.id == region_id:
                return e
            if e.blossom_children:
                queue.extend(e.blossom_children)

        raise ValueError(f'No region with id {region_id}')

    def region_pair_to_line_segment_at_time(
        self, region1: int, region2: int, time: int
    ) -> Tuple[complex, complex]:
        r1 = self._find_potentially_inactive_region(region1)
        r2 = self._find_potentially_inactive_region(region2)
        x, y = min(
            [(a, b) for a in r1.iter_all_sources() for b in r2.iter_all_sources()],
            key=lambda e: abs(e[0] - e[1]),
        )
        return x, y

    def region_boundary_pair_to_line_segment_at_time(
        self, region1: int, boundary: Any, time: int
    ) -> Tuple[complex, complex]:
        r1 = self._find_potentially_inactive_region(region1)
        x, y = min(
            [(a, boundary) for a in r1.iter_all_sources()], key=lambda e: abs(e[0] - boundary),
        )
        return x, y

    def draw(self, *, screen, scale: float):
        # coverage: ignore
        for region in self._region_data_map.values():
            self._draw_region(region=region, screen=screen, scale=scale)

    def _draw_region(
        self,
        *,
        region: GraphFillRegion,
        scale: float,
        screen,
        color: Optional[Tuple[int, int, int]] = None,
        pad: Union[Varying, float] = 0,
        no_second_recurse: bool = False,
    ):
        # coverage: ignore
        if color is None:
            if region.radius.slope == 0:
                color = (0, 255, 0)
            elif region.radius.slope > 0:
                color = (255, 0, 0)
            else:
                color = (255, 255, 0)

        if region.source is not None:
            self._draw_node_radius(
                node=region.source,
                max_radius=(pad + region.radius)(self.time),
                screen=screen,
                color=color,
                scale=scale,
            )

        if region.blossom_children:
            # Fill in blossom area.
            for child in region.blossom_children:
                self._draw_region(
                    region=child,
                    screen=screen,
                    color=color,
                    scale=scale,
                    pad=pad + region.radius,
                    no_second_recurse=True,
                )

            # Draw child regions darker within the blossom area.
            r, g, b = color
            darken = 0.75
            color = int(r * darken), int(g * darken), int(b * darken)
            if not no_second_recurse:
                for child in region.blossom_children:
                    self._draw_region(
                        region=child,
                        screen=screen,
                        color=color,
                        scale=scale,
                        no_second_recurse=no_second_recurse,
                    )

    def _draw_node_radius(
        self,
        *,
        node: TLocation,
        max_radius: float,
        screen,
        color: Tuple[int, int, int],
        scale: float,
    ):
        # coverage: ignore
        import pygame

        heap = [(0, 0, node)]
        seen = set()
        k = 0
        while heap:
            d, _, n = heapq.heappop(heap)
            if n in seen:
                continue
            seen.add(n)

            p0 = complex(n)
            x0, y0 = int(p0.real * scale + 0.5), int(p0.imag * scale + 0.5)
            for distance, neighbor in self.neighbors(n):
                if neighbor in seen:
                    continue
                delta = complex(neighbor) - complex(n)
                p1 = p0 + delta * min(1, (max_radius - d) / distance)
                p2 = p0 + delta * min(1, (max_radius - d) / distance) * (1 + 5 / abs(delta) / scale)
                x1, y1 = int(p1.real * scale + 0.5), int(p1.imag * scale + 0.5)
                x2, y2 = int(p2.real * scale + 0.5), int(p2.imag * scale + 0.5)
                if d + distance < max_radius:
                    heapq.heappush(heap, (d + distance, k, neighbor))

                pygame.draw.line(
                    screen,
                    (0, 0, 0),
                    (int(p0.real * scale + 0.5), int(p0.imag * scale + 0.5)),
                    (x2, y2),
                    3,
                )

                pygame.draw.line(
                    screen, color, (x0, y0), (x1, y1), 3,
                )

                k += 1
