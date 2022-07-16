import dataclasses
from typing import Dict, Generic, Optional, List, TypeVar, TYPE_CHECKING, \
    Iterator, Tuple, Any
import math

import networkx as nx

from slowmatch.events import TentativeEvent, TentativeNeighborInteractionEvent
from slowmatch.varying import Varying

if TYPE_CHECKING:
    from slowmatch.graph_fill_region import GraphFillRegion
    import pygame

TLocation = TypeVar('TLocation')


@dataclasses.dataclass
class Graph:
    nodes: Dict[TLocation, 'DetectorNode'] = dataclasses.field(default_factory=lambda: dict())
    num_observables: int = 0

    def add_edge(self, u: TLocation, v: TLocation, weight: int, observables: int):
        """Add an edge to the graph."""
        if u not in self.nodes:
            self.nodes[u] = DetectorNode(loc=u)
        if v not in self.nodes:
            self.nodes[v] = DetectorNode(loc=v)

        udata = self.nodes[u]
        vdata = self.nodes[v]

        udata.neighbors.append(vdata)
        udata.neighbor_distances.append(weight)
        udata.neighbor_observables.append(observables)
        udata.neighbor_schedule_list.append(None)
        udata.neighbor_back_index.append(len(vdata.neighbors))

        vdata.neighbors.append(udata)
        vdata.neighbor_distances.append(weight)
        vdata.neighbor_observables.append(observables)
        vdata.neighbor_schedule_list.append(None)
        vdata.neighbor_back_index.append(len(udata.neighbors) - 1)

        num_obs_bits = len(bin(observables)[:1:-1])
        if num_obs_bits > self.num_observables:
            self.num_observables = num_obs_bits

    def add_boundary_edge(self, u: TLocation, weight: int, observables: int):
        if u not in self.nodes:
            self.nodes[u] = DetectorNode(loc=u)

        udata = self.nodes[u]
        udata.neighbors.append(None)
        udata.neighbor_distances.append(weight)
        udata.neighbor_observables.append(observables)
        udata.neighbor_schedule_list.append(None)
        udata.neighbor_back_index.append(None)

    def iter_all_edges(self) -> Iterator[Tuple['DetectorNode', int]]:
        seen_nodes = set()
        seen_edges = set()
        for node in self.nodes.values():
            if node.loc not in seen_nodes:
                seen_edges.add(node.loc)
                for i in range(len(node.neighbors)):
                    if node.neighbors[i] is not None:
                        loc2 = node.neighbors[i].loc
                        if (node.loc, loc2) not in seen_edges:
                            seen_edges.add((node.loc, loc2))
                            seen_edges.add((loc2, node.loc))
                            yield node, i

    def draw(self, screen: 'pygame.Screen', scale: float):
        import pygame
        for node, i in self.iter_all_edges():
            v = node.neighbors[i]
            pygame.draw.line(
                screen,
                (150, 150, 150),
                (int(node.loc.real * scale),
                 int(node.loc.imag * scale)),
                (int(v.loc.real * scale),
                 int(v.loc.imag * scale)),
                width=1,
            )


class DetectorNode(Generic[TLocation]):
    """A node in the detector graph.

    Corresponds to a comparison performed to check for errors. A potential
    location where detection events can occur. Edges between these nodes
    correspond to physical errors.
    """

    def __init__(self, loc: TLocation) -> None:
        self.loc = loc
        self.observables_crossed: int = 0
        self.reached_from_source: Optional[DetectorNode] = None
        self.distance_from_source: Optional[int] = None
        self.region_that_arrived: Optional['GraphFillRegion'] = None

        # Reference to each neighbor node, for each neighbor.
        self.neighbors: List[Optional['DetectorNode']] = []
        # Distance across edge to neighbor, for each neighbor.
        self.neighbor_distances: List[int] = []
        # Observables crossed by following edge to neighbor, for each neighbor.
        self.neighbor_observables: List[int] = []
        # Reference to tentative interaction event involving other node, or None, for each neighbor.
        self.neighbor_schedule_list: List[Optional[TentativeNeighborInteractionEvent]] = []
        # Index to self in the other node, for each neighbor.
        self.neighbor_back_index: List[Optional[int]] = []

        # Temporary state used during dijkstra search.
        self.distance_from_search_source: Optional[int] = None
        # Temporary state used during dijkstra search.
        self.search_predecessor: Optional[int] = None

    @property
    def num_neighbors(self) -> int:
        return len(self.neighbors)

    def has_same_owner_as(self, other: 'DetectorNode') -> bool:
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
        for e in self.neighbor_schedule_list:
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
        self.neighbor_schedule_list = [None] * len(self.neighbors)

    def top_region(self) -> Optional['GraphFillRegion']:
        if self.region_that_arrived is None:
            return None
        return self.region_that_arrived.top_region()

    def in_active_region(self):
        if self.region_that_arrived is not None:
            return self.region_that_arrived.blossom_parent is None
        return False

    def distance_from_source_almost_reached_from(self, source: 'DetectorNode') -> int:
        min_d = math.inf
        for i in range(len(self.neighbors)):
            v = self.neighbors[i]
            if v is not None:
                if v.reached_from_source is source:
                    d = v.distance_from_source + self.neighbor_distances[i]
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
            )
        elif ub and not vb:
            new_graph.add_boundary_edge(
                u=v,
                weight=w,
                observables=obs,
            )
    return new_graph
