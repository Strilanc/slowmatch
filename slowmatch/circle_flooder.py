import dataclasses
import math
from typing import TypeVar, List, Tuple, Optional, Dict, Union, Iterator

import cirq

from slowmatch import Varying
from slowmatch.flooder import MwpmEvent, Flooder, RegionHitRegionEvent, BlossomImplodeEvent

TLocation = TypeVar('TLocation')


@cirq.value_equality(approximate=True)
class VaryingCircle:
    def __init__(self,
                 *,
                 id: int,
                 center: Union[int, float, complex],
                 radius: Union[int, float, Varying]):
        self.id = id
        self.center = complex(center)
        self.radius = Varying(radius)

    def _value_equality_values_(self):
        return self.id, self.center, self.radius

    def iter_all_circles(self) -> Iterator['VaryingCircle']:
        yield self

    def __add__(self, other: Varying) -> 'VaryingCircle':
        return VaryingCircle(center=self.center,
                             id=self.id,
                             radius=self.radius + other)

    def implosion_time(self) -> Optional[float]:
        if self.radius.slope >= 0:
            return None
        return self.radius.zero_intercept()

    def distance_from_at(self, other: Union['VaryingCircleBlossom', 'VaryingCircle'], time: float) -> float:
        assert isinstance(time, (int, float))
        if isinstance(other, VaryingCircleBlossom):
            return other.distance_from_at(self, time)
        r0 = self.radius(time)
        r1 = other.radius(time)
        return abs(self.center - other.center) - r1 - r0

    def collision_time(self, other: Union['VaryingCircleBlossom', 'VaryingCircle']) -> Optional[float]:
        if isinstance(other, VaryingCircleBlossom):
            return other.collision_time(self)

        approach_speed = self.radius.slope + other.radius.slope
        if approach_speed <= 0:
            return None
        base_time = self.radius._base_time
        distance = self.distance_from_at(other, base_time)
        return base_time + distance / approach_speed

    def draw(self, *, time: float, screen, color: Optional[Tuple[int, int, int]] = None):
        import pygame
        if color is None:
            if self.radius.slope == 0:
                color = (0, 255, 0)
            elif self.radius.slope > 0:
                color = (255, 0, 0)
            else:
                color = (255, 255, 0)
        x = self.center.real
        y = self.center.imag
        r = self.radius(time)
        pygame.draw.circle(screen, color, (int(x), int(y)), int(math.ceil(r)))

    def __repr__(self):
        return f'VaryingCircle(id={self.id!r}, center={self.center!r}, radius={self.radius!r})'


@dataclasses.dataclass
class VaryingCircleBlossom:
    children: List[Union['VaryingCircleBlossom', VaryingCircle]]
    radius: Varying
    id: int

    def __add__(self, other: Varying) -> 'VaryingCircleBlossom':
        return VaryingCircleBlossom(children=self.children,
                                    id=self.id,
                                    radius=self.radius + other)

    def implosion_time(self) -> Optional[float]:
        if self.radius.slope >= 0:
            return None
        return self.radius.zero_intercept()

    def distance_from_at(self, other: 'VaryingCircleBlossom', time: float) -> float:
        return min(a.distance_from_at(b, time)
                   for a in self.iter_all_circles()
                   for b in other.iter_all_circles())

    def collision_time(self, other: 'VaryingCircleBlossom') -> Optional[float]:
        times = [a.collision_time(b)
                 for a in self.iter_all_circles()
                 for b in other.iter_all_circles()]
        return min([t for t in times if t is not None], default=None)

    def change_growth_rate_at(self, *, time_of_change: float, new_growth: Optional[int] = None):
        self.radius = self.radius.then_slope_at(
            time_of_change=time_of_change,
            new_slope=new_growth)

    def iter_all_circles(self) -> Iterator[VaryingCircle]:
        for child in self.children:
            child += self.radius
            if isinstance(child, VaryingCircle):
                yield child
            else:
                yield from child.iter_all_circles()

    def draw(self, *, time: float, screen, color: Optional[Tuple[int, int, int]] = None):
        if color is None:
            if self.radius.slope == 0:
                color = (0, 255, 0)
            elif self.radius.slope > 0:
                color = (255, 0, 0)
            else:
                color = (255, 255, 0)

        for child in self.iter_all_circles():
            child.draw(time=time, screen=screen, color=color)

        # Child regions darker.
        r, g, b = color
        darken = 0.75
        color = int(r * darken), int(g * darken), int(b * darken)
        for child in self.children:
            child.draw(time=time, screen=screen, color=color)


class CircleFlooder(Flooder[complex]):
    def __init__(self):
        self.regions: Dict[int, Union[VaryingCircleBlossom, VaryingCircle]] = {}
        self._next_region_id = 0
        self.time = 0

    def create_region(self, location: complex) -> int:
        k = self._next_region_id
        self._next_region_id += 1
        self.regions[k] = VaryingCircle(
            center=complex(location),
            id=k,
            radius=Varying(base_time=self.time, slope=1))
        return k

    def set_region_growth(self, region_id: int, *, new_growth: int):
        r = self.regions[region_id]
        r.radius = r.radius.then_slope_at(
            time_of_change=self.time, new_slope=new_growth)

    def create_blossom(self, contained_region_ids: List[int]) -> int:
        regions = []
        for i in contained_region_ids:
            regions.append(self.regions[i])
            del self.regions[i]

        for r in regions:
            r.radius = r.radius.then_slope_at(
                time_of_change=self.time, new_slope=0)

        k = self._next_region_id
        self._next_region_id += 1
        self.regions[k] = VaryingCircleBlossom(
            id=k,
            children=regions,
            radius=Varying(base_time=self.time, slope=1))

        return k

    def _explode_region(self, region_id: int) -> List[int]:
        region = self.regions[region_id]
        assert abs(region.radius(self.time)) < 1e-8
        del self.regions[region_id]
        for child in region.children:
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
            if isinstance(r, VaryingCircleBlossom):
                t = r.implosion_time()
                if t is not None:
                    in_out_touch_pairs = []
                    for child in r.children:
                        for other in self.regions.values():
                            if other.id != r.id and abs(child.distance_from_at(other, t)) < 1e-8:
                                in_out_touch_pairs.append((child.id, other.id))
                    yield BlossomImplodeEvent(
                        blossom_region_id=r.id,
                        in_out_touch_pairs=in_out_touch_pairs,
                        time=t)

    def next_event(self, max_time: float = float('inf')) -> Optional[MwpmEvent]:
        best_event = None
        for event in self._iter_events():
            if best_event is None or event.time < best_event.time:
                best_event = event
        if best_event is None:
            return None
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
            if isinstance(e, VaryingCircleBlossom):
                queue.extend(e.children)

        raise ValueError(f'No region with id {region_id}')

    def region_pair_to_line_segment_at_time(
            self,
            region1: int,
            region2: int,
            time: float) -> Tuple[complex, complex]:
        r1 = self._find_potentially_inactive_region(region1)
        r2 = self._find_potentially_inactive_region(region2)
        x, y = min(
            [
                (a, b)
                for a in r1.iter_all_circles()
                for b in r2.iter_all_circles()
            ],
            key=lambda e: e[0].distance_from_at(e[1], time))
        return x.center, y.center

    def draw(self,
             *,
             screen,
             time_delta: float = 0):
        for region in self.regions.values():
            region.draw(screen=screen, time=self.time + time_delta)
