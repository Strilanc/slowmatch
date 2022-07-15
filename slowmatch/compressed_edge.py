import dataclasses
import heapq
import math
from typing import TYPE_CHECKING, Optional, Tuple, List


if TYPE_CHECKING:
    from slowmatch.graph import LocationData
    import pygame


def join_edges(edges: List['CompressedEdge']) -> 'CompressedEdge':
    x = edges[0]
    for e in edges[1:]:
        x = x & e
    return x


def _path_from_predecessors(node: 'LocationData') -> List['CompressedEdge']:
    edges = []
    while node.search_predecessor is not None:
        next_node = node.neighbors_with_boundary[node.search_predecessor]
        edges.append(
            CompressedEdge(
                source1=node,
                source2=next_node,
                obs_mask = node.observables[node.search_predecessor],
                distance = node.distances[node.search_predecessor]
            )
        )
        node = next_node
    return [e.reversed() for e in reversed(edges)]


def _cleanup_searched_nodes(nodes: List['LocationData']):
    for node in nodes:
        node.search_predecessor = None
        node.distance_from_search_source = None


def _min_edge_to_boundary(node: 'LocationData') -> Tuple[int, int]:
    min_bound_distance = math.inf
    min_bound_obs = None
    for i in range(len(node.neighbors)):
        if node.neighbors_with_boundary[i] is None:
            if node.distances[i] < min_bound_distance:
                min_bound_distance = node.distances[i]
                min_bound_obs = node.observables[i]
    return min_bound_distance, min_bound_obs


@dataclasses.dataclass
class CompressedEdge:
    source1: 'LocationData'
    source2: Optional['LocationData']
    obs_mask: int
    distance: int

    def reversed(self) -> 'CompressedEdge':
        return CompressedEdge(
            source1=self.source2,
            source2=self.source1,
            obs_mask=self.obs_mask,
            distance=self.distance
        )

    def merged_with(self, other: 'CompressedEdge'):
        if self.source2 is not other.source1:
            raise ValueError("self.source2 must be other.source1 to merge self with other")
        assert self.source2 is other.source1
        return CompressedEdge(
            source1=self.source1,
            source2=other.source2,
            obs_mask=self.obs_mask ^ other.obs_mask,
            distance=self.distance + other.distance
        )

    def __and__(self, other: 'CompressedEdge') -> 'CompressedEdge':
        return self.merged_with(other)

    def __eq__(self, other: 'CompressedEdge') -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.source1 == other.source1 and
            self.source2 == other.source2 and
            self.obs_mask == other.obs_mask and
            self.distance == other.distance
        )

    def __str__(self):
        return f"(source1={self.source1}, source2={self.source2}, obs_mask={self.obs_mask}, distance={self.distance})"

    def to_path(self) -> List['CompressedEdge']:
        if self.source1 is None and self.source2 is None:
            raise ValueError(f"Both source1 and source2 are None")
        elif self.source1 is None:
            return list(reversed([e.reversed() for e in self.reversed().to_path()]))
        explored = []
        queue: List[Tuple[int, int, 'LocationData']] = []
        current_node = self.source1
        current_node.distance_from_search_source = 0
        heapq.heappush(queue, (0, 0, current_node))
        heap_counter = 1
        explored.append(current_node)
        boundary_distance = None
        boundary_predecessor = None
        res = None
        while heapq:
            distance, _, current_node = heapq.heappop(queue)
            if current_node is not None and distance != current_node.distance_from_search_source:
                continue
            elif current_node is None and distance != boundary_distance:
                continue
            if current_node is None and self.source2 is None:
                # Found boundary (self.source2)
                min_bound_distance, min_bound_obs = _min_edge_to_boundary(boundary_predecessor)
                res = _path_from_predecessors(boundary_predecessor)
                res.append(CompressedEdge(source1=boundary_predecessor, source2=None,
                                          obs_mask=min_bound_obs, distance=min_bound_distance))
                break
            elif current_node is self.source2 and current_node is not None:
                # Found self.source2
                assert distance == self.distance
                res = _path_from_predecessors(current_node)
                break
            for i in range(len(current_node.neighbors)):
                v = current_node.neighbors_with_boundary[i]
                new_distance = distance + current_node.distances[i]
                if v is not None and v.distance_from_search_source is None:
                    explored.append(v)
                if v is None:
                    if self.source2 is not None:
                        continue
                    else:
                        if boundary_distance is None or new_distance < boundary_distance:
                            boundary_distance = new_distance
                            boundary_predecessor = current_node
                elif v.distance_from_search_source is None or new_distance < v.distance_from_search_source:
                    v.distance_from_search_source = new_distance
                    v.search_predecessor = current_node.neighbor_index[i]
                else:
                    continue
                heap_counter += 1
                heapq.heappush(queue, (new_distance, heap_counter, v))
        _cleanup_searched_nodes(explored)
        if res is not None:
            return res
        src2 = self.source2.loc if self.source2 is not None else None
        raise ValueError(f"No path found from {self.source1.loc} to {src2}")

    def draw(self, screen: 'pygame.Surface', scale: float, rgb: Tuple[int, int, int], width: int = 4) -> None:
        """Draws the compressed edge, provided LocationData.loc is of type `complex'"""
        import pygame
        if self.source1 is not None and self.source2 is not None:
            pygame.draw.line(
                screen,
                rgb,
                (int(self.source1.loc.real * scale), int(self.source1.loc.imag * scale)),
                (int(self.source2.loc.real * scale), int(self.source2.loc.imag * scale)),
                width=width,
            )

    def draw_path(self, screen: 'pygame.Surface', scale: float, rgb: Tuple[int, int, int], width: int = 4):
        for edge in self.to_path():
            edge.draw(screen=screen, scale=scale, rgb=rgb, width=width)
