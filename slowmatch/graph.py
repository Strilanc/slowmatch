import dataclasses
from typing import Dict, Generic, Optional, List, TypeVar, TYPE_CHECKING, \
    Iterator, Tuple, Any
import math

import networkx as nx

from slowmatch.events import TentativeEvent
from slowmatch.varying import Varying

if TYPE_CHECKING:
    from slowmatch.graph_fill_region import GraphFillRegion
    import pygame

TLocation = TypeVar('TLocation')


@dataclasses.dataclass
class Graph:
    nodes: Dict[TLocation, 'LocationData'] = dataclasses.field(default_factory=lambda: dict())
    num_observables: int = 0

    def add_edge(self, u: TLocation, v: TLocation, weight: int, observables: int):
        """Add an edge to the graph."""
        if u not in self.nodes:
            self.nodes[u] = LocationData(loc=u)
        if v not in self.nodes:
            self.nodes[v] = LocationData(loc=v)

        udata = self.nodes[u]
        vdata = self.nodes[v]

        udata.neighbors.append(vdata)
        udata.neighbors_with_boundary.append(vdata)
        udata.distances.append(weight)
        udata.observables.append(observables)
        udata.schedule_list.append(None)
        udata.neighbor_index.append(len(vdata.neighbors))

        vdata.neighbors.append(udata)
        vdata.neighbors_with_boundary.append(udata)
        vdata.distances.append(weight)
        vdata.observables.append(observables)
        vdata.schedule_list.append(None)
        vdata.neighbor_index.append(len(udata.neighbors) - 1)

        num_obs_bits = len(bin(observables)[:1:-1])
        if num_obs_bits > self.num_observables:
            self.num_observables = num_obs_bits

    def add_boundary_edge(self, u: TLocation, weight: int, observables: int,
                          boundary_node: Optional[TLocation] = None):
        if u not in self.nodes:
            self.nodes[u] = LocationData(loc=u)
        if boundary_node is not None:
            if boundary_node not in self.nodes:
                self.nodes[boundary_node] = LocationData(loc=boundary_node)
            vdata = self.nodes[boundary_node]
        else:
            vdata = None

        udata = self.nodes[u]
        udata.neighbors.append(None)
        udata.neighbors_with_boundary.append(vdata)
        udata.distances.append(weight)
        udata.observables.append(observables)
        udata.schedule_list.append(None)
        vind = len(vdata.neighbors) if vdata is not None else None
        udata.neighbor_index.append(vind)

        # For the demo, it can be useful to have boundary nodes
        if vdata is not None:
            vdata.neighbors.append(udata)
            vdata.neighbors_with_boundary.append(udata)
            vdata.neighbor_index.append(len(udata.neighbors) - 1)
            vdata.distances.append(weight)
            vdata.observables.append(observables)
            vdata.schedule_list.append(None)

    def iter_all_edges(self) -> Iterator[Tuple['LocationData', int]]:
        seen_nodes = set()
        seen_edges = set()
        for node in self.nodes.values():
            if node.loc not in seen_nodes:
                seen_edges.add(node.loc)
                for i in range(len(node.neighbors)):
                    loc2 = node.neighbors_with_boundary[i].loc
                    if (node.loc, loc2) not in seen_edges:
                        seen_edges.add((node.loc, loc2))
                        seen_edges.add((loc2, node.loc))
                        yield node, i

    def draw(self, screen: 'pygame.Screen', scale: float):
        import pygame
        for node, i in self.iter_all_edges():
            v = node.neighbors_with_boundary[i]
            pygame.draw.line(
                screen,
                (150, 150, 150),
                (int(node.loc.real * scale),
                 int(node.loc.imag * scale)),
                (int(v.loc.real * scale),
                 int(v.loc.imag * scale)),
                width=1,
            )


class LocationData(Generic[TLocation]):
    def __init__(self, loc: TLocation) -> None:
        self.loc = loc
        self.observables_crossed: int = 0
        self.reached_from_source: Optional[LocationData] = None
        self.distance_from_source: Optional[int] = None
        self.region_that_arrived: Optional['GraphFillRegion'] = None
        self.neighbors: List[Optional['LocationData']] = []
        self.neighbors_with_boundary: List['LocationData'] = []
        self.distances: List[int] = []
        self.observables: List[int] = []
        self.schedule_list: List[Optional[TentativeEvent]] = []
        self.neighbor_index: List[Optional[int]] = []
        self.distance_from_search_source: Optional[int] = None
        self.search_predecessor: Optional[int] = None

    @property
    def num_neighbors(self) -> int:
        return len(self.neighbors)

    def has_same_owner_as(self, other: 'LocationData') -> bool:
        if not self.region_that_arrived or not other.region_that_arrived:
            return False
        return self.top_region() is other.top_region()

    def is_owned_by(self, region: 'GraphFillRegion') -> bool:
        if self.region_that_arrived is None:
            return False
        curr_region = self.region_that_arrived
        while curr_region.blossom_parent is not None:
            if curr_region is region:
                return True
            curr_region = curr_region.blossom_parent
        return curr_region is region

    def cumulative_radius_to_region(self, region: 'GraphFillRegion') -> Varying:
        if not self.is_owned_by(region):
            raise ValueError(f"{self} not owned by {region}")
        curr_region = self.region_that_arrived
        radius = curr_region.radius
        while curr_region.blossom_parent is not None and curr_region is not region:
            curr_region = curr_region.blossom_parent
            radius += curr_region.radius
        assert curr_region is region
        return radius

    def invalidate_involved_schedule_items(self):
        for e in self.schedule_list:
            if e is not None:
                e.invalidate()

    def local_radius(self) -> Varying:
        if self.region_that_arrived is not None:
            return self.total_radius() - self.distance_from_source
        return Varying(0)

    def total_radius(self) -> Varying:
        src = self.reached_from_source
        if not src:
            return Varying(0)
        cur_region = src.region_that_arrived
        tot_rad = cur_region.radius
        while cur_region.blossom_parent is not None:
            cur_region = cur_region.blossom_parent
            tot_rad += cur_region.radius
        return tot_rad

    def is_empty(self) -> bool:
        return self.reached_from_source is None

    def cleanup(self) -> None:
        self.observables_crossed = 0
        self.reached_from_source = None
        self.distance_from_source = None
        self.region_that_arrived = None
        self.schedule_list = [None] * len(self.neighbors)

    def top_region(self) -> Optional['GraphFillRegion']:
        if self.region_that_arrived is None:
            return None
        return self.region_that_arrived.top_region()

    def in_active_region(self):
        if self.region_that_arrived is not None:
            return self.region_that_arrived.blossom_parent is None
        return False

    def distance_from_source_almost_reached_from(self, source: 'LocationData') -> int:
        min_d = math.inf
        for i in range(len(self.neighbors)):
            v = self.neighbors_with_boundary[i]
            if v.reached_from_source is source:
                d = v.distance_from_source + self.distances[i]
                min_d = min(d, min_d)
        if min_d == math.inf:
            raise ValueError(f"No neighbouring nodes have been reached from {source}")
        return min_d

    def draw(self, screen: 'pygame.Surface', scale: float) -> None:
        import pygame
        pygame.draw.circle(surface=screen, color=(255, 0, 255),
                           center=((self.loc*scale).real, (self.loc*scale).imag),
                           radius=5)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return self.loc == other.loc
        return False

    def __str__(self) -> str:
        return str(self.loc)


def graph_from_networkx(graph: nx.Graph) -> Graph:
    new_graph = Graph()
    for u, v, data in graph.edges(data=True):
        ub = graph.nodes[u].get("is_boundary", False) or "boundary" in graph.nodes[u]
        vb = graph.nodes[v].get("is_boundary", False) or "boundary" in graph.nodes[v]
        w = int(data.get('weight', 1))
        obs = int(data.get('observables', 0))
        if not ub and not vb:
            new_graph.add_edge(
                u=u,
                v=v,
                weight=w,
                observables=obs
            )
        elif not ub and vb:
            new_graph.add_boundary_edge(
                u=u,
                weight=w,
                observables=obs,
                boundary_node=v
            )
        elif ub and not vb:
            new_graph.add_boundary_edge(
                u=v,
                weight=w,
                observables=obs,
                boundary_node=u
            )
    return new_graph
