import dataclasses
import math
from typing import TypeVar, List, Tuple, Optional, Dict, Union, Iterator

from slowmatch.fill_system import VaryingRadius, Event, FillSystem, RegionHitRegionEvent, BlossomImplodeEvent

TLocation = TypeVar('TLocation')


@dataclasses.dataclass
class GrowingCircle:
    center: complex
    growing_radius: VaryingRadius
    id: int

    def iter_all_circles(self) -> Iterator['GrowingCircle']:
        yield self

    def __add__(self, other: VaryingRadius) -> 'GrowingCircle':
        return GrowingCircle(center=self.center,
                             id=self.id,
                             growing_radius=self.growing_radius + other)

    def implosion_time(self) -> Optional[float]:
        speed = self.growing_radius.growth
        if self.growing_radius.growth >= 0:
            return None
        return self.growing_radius.time0 - self.growing_radius.radius0 / speed

    def collision_time(self, other: Union['GrowingCircleTree', 'GrowingCircle']) -> Optional[float]:
        if isinstance(other, GrowingCircleTree):
            return other.collision_time(self)

        approach_speed = self.growing_radius.growth + other.growing_radius.growth
        if approach_speed <= 0:
            return None
        base_time = self.growing_radius.time0
        r0 = self.growing_radius.radius_at(base_time)
        r1 = other.growing_radius.radius_at(base_time)
        distance = abs(self.center - other.center) - r1 - r0
        return base_time + distance / approach_speed

    def draw(self, *, time: float, screen, color: Optional[Tuple[int, int, int]] = None):
        import pygame
        if color is None:
            if self.growing_radius.growth == 0:
                color = (0, 255, 0)
            elif self.growing_radius.growth > 0:
                color = (255, 0, 0)
            else:
                color = (255, 255, 0)
        x = self.center.real
        y = self.center.imag
        r = self.growing_radius.radius_at(time)
        pygame.draw.circle(screen, color, (int(x), int(y)), int(math.ceil(r)))


@dataclasses.dataclass
class GrowingCircleTree:
    children: Union['GrowingCircleTree', GrowingCircle]
    growing_radius: VaryingRadius
    id: int

    def __add__(self, other: VaryingRadius) -> 'GrowingCircleTree':
        return GrowingCircleTree(children=self.children,
                                 id=self.id,
                                 growing_radius=self.growing_radius + other)

    def implosion_time(self) -> Optional[float]:
        speed = self.growing_radius.growth
        if self.growing_radius.growth >= 0:
            return None
        return self.growing_radius.time0 - self.growing_radius.radius0 / speed

    def collision_time(self, other: 'GrowingCircleTree') -> Optional[float]:
        times = [a.collision_time(b)
                 for a in self.iter_all_circles()
                 for b in other.iter_all_circles()]
        return min([t for t in times if t is not None], default=None)

    def sync(self, new_time: float, new_growth: Optional[int] = None):
        self.growing_radius.sync(new_time, new_growth)

    def iter_all_circles(self) -> Iterator[GrowingCircle]:
        for child in self.children:
            child += self.growing_radius
            if isinstance(child, GrowingCircle):
                yield child
            else:
                yield from child.iter_all_circles()

    def draw(self, *, time: float, screen, color: Optional[Tuple[int, int, int]] = None):
        if color is None:
            if self.growing_radius.growth == 0:
                color = (0, 255, 0)
            elif self.growing_radius.growth > 0:
                color = (255, 0, 0)
            else:
                color = (255, 255, 0)

        for child in self.iter_all_circles():
            child.draw(time=time, screen=screen, color=color)

        # Child regions darker.
        r, g, b = color
        color = int(r) * 0.9, int(g) * 0.9, int(b) * 0.9
        for child in self.children:
            child.draw(time=time, screen=screen, color=color)


class InefficientEuclideanFillSystem(FillSystem[complex]):
    def __init__(self):
        self.regions: Dict[int, Union[GrowingCircleTree, GrowingCircle]] = {}
        self._next_region_id = 0
        self.time = 0

    def get_region_growth(self, region_id: int) -> int:
        return self.regions[region_id].growing_radius.growth

    def create_region(self, location: complex) -> int:
        k = self._next_region_id
        self._next_region_id += 1
        self.regions[k] = GrowingCircle(center=complex(location),
                                        id=k,
                                        growing_radius=VaryingRadius(time0=self.time))

    def set_region_growth(self, region_id: int, *, new_growth: int):
        self.regions[region_id].growing_radius.sync(self.time, new_growth=new_growth)

    def create_blossom(self, contained_region_ids: List[int]) -> int:
        regions = []
        for i in contained_region_ids:
            regions.append(self.regions[i])
            del self.regions[i]

        for r in regions:
            r.growing_radius.sync(new_time=self.time, new_growth=0)

        k = self._next_region_id
        self._next_region_id += 1
        self.regions[k] = GrowingCircleTree(
            id=k,
            children=regions,
            growing_radius=VaryingRadius(time0=self.time))

        return k

    def explode_region(self, region_id: int) -> List[int]:
        region = self.regions[region_id]
        assert region.growing_radius.radius_at(self.time) == 0
        del self.regions[region_id]
        for child in region.children:
            assert child.id not in self.regions
            self.regions[child.id] = child

    def _iter_events(self) -> Iterator[Event]:
        regions = sorted(self.regions.values(), key=lambda e: e.id)
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                r1 = regions[i]
                r2 = regions[j]
                t = r1.collision_time(r2)
                if t is not None:
                    yield RegionHitRegionEvent(region1=r1.id, region2=r2.id, time=t)

        for r in self.regions.values():
            t = r.implosion_time()
            if t is not None:
                yield BlossomImplodeEvent(region1=r.id, region2=None, time=t)

    def next_event(self, max_time: Optional[float] = None) -> Event:
        best_event = None
        for event in self._iter_events():
            if best_event is None or event.time < best_event.time:
                best_event = event
        if best_event is None:
            return None
        if max_time is not None and best_event.time > max_time:
            self.time = max_time
            return None
        self.time = best_event.time
        return best_event

    def draw(self, *, screen, time_delta: float = 0):
        for region in self.regions.values():
            region.draw(screen=screen, time=self.time + time_delta)
