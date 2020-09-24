import dataclasses
import math
from typing import TypeVar, List, Tuple, Optional, Dict, Union, Iterator, Any

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


@cirq.value_equality(approximate=True)
class CircleFillRegion:
    def __init__(
        self,
        *,
        id: int,
        radius: Union[int, float, Varying],
        source: Union[None, int, float, complex] = None,
        blossom_children: Optional[List['CircleFillRegion']] = None,
    ):
        self.id = id
        self.source = None if source is None else complex(source)
        self.radius = Varying(radius)
        self.blossom_children = blossom_children

    def _value_equality_values_(self):
        return self.id, self.source, self.radius, self.blossom_children

    def iter_all_circles(self) -> Iterator['CircleFillRegion']:
        if self.blossom_children is not None:
            for child in self.blossom_children:
                yield from (child + self.radius).iter_all_circles()
        if self.source is not None:
            yield self

    def __add__(self, other: Varying) -> 'CircleFillRegion':
        return CircleFillRegion(
            source=self.source,
            blossom_children=self.blossom_children,
            id=self.id,
            radius=self.radius + other,
        )

    def implosion_time(self) -> Optional[float]:
        if self.radius.slope >= 0:
            return None
        return self.radius.zero_intercept()

    def distance_from_at(self, other: 'CircleFillRegion', time: float) -> float:
        if self.source is None or other.source is None:
            return min(
                a.distance_from_at(b, time)
                for a in self.iter_all_circles()
                for b in other.iter_all_circles()
            )
        assert isinstance(time, (int, float))
        r0 = self.radius(time)
        r1 = other.radius(time)
        return abs(self.source - other.source) - r1 - r0

    def collision_time(self, other: 'CircleFillRegion') -> Optional[float]:
        if self.source is None or other.source is None:
            times = [
                a.collision_time(b)
                for a in self.iter_all_circles()
                for b in other.iter_all_circles()
            ]
            return min([t for t in times if t is not None], default=None)

        approach_speed = self.radius.slope + other.radius.slope
        if approach_speed <= 0:
            return None
        base_time = self.radius._base_time
        distance = self.distance_from_at(other, base_time)
        return base_time + distance / approach_speed

    def draw(
        self, *, time: float, screen, color: Optional[Tuple[int, int, int]] = None, scale: float = 1
    ):
        # coverage: ignore
        import pygame

        if color is None:
            if self.radius.slope == 0:
                color = (0, 255, 0)
            elif self.radius.slope > 0:
                color = (255, 0, 0)
            else:
                color = (255, 255, 0)

        if self.source is not None:
            x = self.source.real * scale
            y = self.source.imag * scale
            r = self.radius(time) * scale
            pygame.draw.circle(screen, color, (int(x), int(y)), int(math.ceil(r)))
        else:
            for child in self.iter_all_circles():
                child.draw(time=time, screen=screen, color=color, scale=scale)

            # Child regions darker.
            r, g, b = color
            darken = 0.75
            color = int(r * darken), int(g * darken), int(b * darken)
            for child in self.blossom_children:
                child.draw(time=time, screen=screen, color=color, scale=scale)

    def __repr__(self):
        return f'VaryingCircle(id={self.id!r}, center={self.source!r}, radius={self.radius!r})'


class CircleFlooder(Flooder[complex]):
    def __init__(self, boundary_radius: Optional[float] = None):
        self.regions: Dict[int, CircleFillRegion] = {}
        self._next_region_id = 0
        self.time = 0
        self.boundary_radius = boundary_radius

    def create_region(self, location: complex) -> int:
        k = self._next_region_id
        self._next_region_id += 1
        self.regions[k] = CircleFillRegion(
            source=complex(location), id=k, radius=Varying(base_time=self.time, slope=1)
        )
        return k

    def set_region_growth(self, region_id: int, *, new_growth: int):
        r = self.regions[region_id]
        r.radius = r.radius.then_slope_at(time_of_change=self.time, new_slope=new_growth)

    def create_blossom(self, contained_region_ids: List[int]) -> int:
        regions = []
        for i in contained_region_ids:
            regions.append(self.regions[i])
            del self.regions[i]

        for r in regions:
            r.radius = r.radius.then_slope_at(time_of_change=self.time, new_slope=0)

        k = self._next_region_id
        self._next_region_id += 1
        self.regions[k] = CircleFillRegion(
            id=k, blossom_children=regions, radius=Varying(base_time=self.time, slope=1)
        )

        return k

    def _explode_region(self, region_id: int) -> List[int]:
        region = self.regions[region_id]
        assert abs(region.radius(self.time)) < 1e-8
        del self.regions[region_id]
        for child in region.blossom_children:
            assert child.id not in self.regions
            self.regions[child.id] = child

    def _iter_events(self) -> Iterator[MwpmEvent]:
        regions = sorted(self.regions.values(), key=lambda e: e.id)
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                r1 = regions[i]
                r2 = regions[j]
                t = r1.collision_time(r2)
                if t is not None:
                    yield RegionHitRegionEvent(region1=r1.id, region2=r2.id, time=t)

        for r in self.regions.values():
            if r.blossom_children is not None:
                t = r.implosion_time()
                if t is not None:
                    in_out_touch_pairs = []
                    for child in r.blossom_children:
                        for other in self.regions.values():
                            if other.id != r.id and abs(child.distance_from_at(other, t)) < 1e-8:
                                in_out_touch_pairs.append((child.id, other.id))
                    yield BlossomImplodeEvent(
                        blossom_region_id=r.id, in_out_touch_pairs=in_out_touch_pairs, time=t
                    )

        if self.boundary_radius is not None:
            for r in self.regions.values():
                t = min(
                    [
                        (self.boundary_radius - c.radius - abs(c.source)).zero_intercept()
                        for c in r.iter_all_circles()
                        if c.radius.slope > 0
                    ],
                    default=None,
                )
                if t is not None:
                    yield RegionHitBoundaryEvent(max(self.time, t), r.id, 'perimeter')

    def next_event(self, max_time: float = float('inf')) -> Optional[MwpmEvent]:
        best_event = None
        for event in self._iter_events():
            if best_event is None or event.time < best_event.time:
                best_event = event
        if best_event is None:
            return None
        assert best_event.time >= self.time
        if best_event.time > max_time:
            self.time = max_time
            return None
        self.time = best_event.time

        if isinstance(best_event, BlossomImplodeEvent):
            self._explode_region(best_event.blossom_region_id)

        return best_event

    def _find_potentially_inactive_region(self, region_id: int):
        if region_id in self.regions:
            return self.regions[region_id]

        queue = [*self.regions.values()]
        while queue:
            e = queue.pop()
            if e.id == region_id:
                return e
            if e.blossom_children is not None:
                queue.extend(e.children)

        raise ValueError(f'No region with id {region_id}')

    def region_pair_to_line_segment_at_time(
        self, region1: int, region2: int, time: float
    ) -> Tuple[complex, complex]:
        r1 = self._find_potentially_inactive_region(region1)
        r2 = self._find_potentially_inactive_region(region2)
        x, y = min(
            [(a, b) for a in r1.iter_all_circles() for b in r2.iter_all_circles()],
            key=lambda e: e[0].distance_from_at(e[1], time),
        )
        return x.source, y.source

    def region_boundary_pair_to_line_segment_at_time(
        self, region1: int, boundary: Any, time: float
    ) -> Tuple[complex, complex]:
        r1 = self._find_potentially_inactive_region(region1)
        c: complex = min(
            [a.source for a in r1.iter_all_circles()],
            key=lambda e: abs(abs(e) - self.boundary_radius),
        )
        d = c / abs(c) * self.boundary_radius
        return c, d

    def draw(self, *, screen, time_delta: float = 0, scale: float = 1):
        # coverage: ignore
        import pygame

        if self.boundary_radius is not None:
            screen.fill((0, 0, 0))
            pygame.draw.circle(
                screen, (255, 255, 255), (0, 0), int(self.boundary_radius * scale) - 2
            )
        for region in self.regions.values():
            region.draw(screen=screen, time=self.time + time_delta, scale=scale)
