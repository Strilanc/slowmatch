import dataclasses
import heapq
import math
from typing import TYPE_CHECKING, Optional, Tuple, List


if TYPE_CHECKING:
    from slowmatch.graph import DetectorNode
    import pygame


def join_edges(edges: List['CompressedEdge']) -> 'CompressedEdge':
    x = edges[0]
    for e in edges[1:]:
        x = x & e
    return x


def _path_from_predecessors(node: 'DetectorNode') -> List['CompressedEdge']:
    edges = []
    while node.search_predecessor is not None:
        next_node = node.neighbors_with_boundary[node.search_predecessor]
        edges.append(
            CompressedEdge(
                loc_from=node,
                loc_to=next_node,
                obs_mask = node.observables[node.search_predecessor],
                distance = node.distances[node.search_predecessor]
            )
        )
        node = next_node
    return [e.reversed() for e in reversed(edges)]


def _cleanup_searched_nodes(nodes: List['DetectorNode']):
    for node in nodes:
        node.search_predecessor = None
        node.distance_from_search_source = None


def _min_edge_to_boundary(node: 'DetectorNode') -> Tuple[int, int]:
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
    """A compressed edge summarizes a path through the detector graph.

    The path is summarized as:
        - The observables that were crossed.
        - The distance that was crossed.
        - The endpoint(s).
    """
    loc_from: 'DetectorNode'
    loc_to: Optional['DetectorNode']
    obs_mask: int
    distance: int

    def reversed(self) -> 'CompressedEdge':
        return CompressedEdge(
            loc_from=self.loc_to,
            loc_to=self.loc_from,
            obs_mask=self.obs_mask,
            distance=self.distance
        )

    def merged_with(self, other: 'CompressedEdge'):
        if self.loc_to is not other.loc_from:
            raise ValueError("self.loc_to must be other.loc_from to merge self with other")
        assert self.loc_to is other.loc_from
        return CompressedEdge(
            loc_from=self.loc_from,
            loc_to=other.loc_to,
            obs_mask=self.obs_mask ^ other.obs_mask,
            distance=self.distance + other.distance
        )

    def __and__(self, other: 'CompressedEdge') -> 'CompressedEdge':
        return self.merged_with(other)

    def __eq__(self, other: 'CompressedEdge') -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
                self.loc_from == other.loc_from and
                self.loc_to == other.loc_to and
                self.obs_mask == other.obs_mask and
                self.distance == other.distance
        )

    def __str__(self) -> str:
        return f"(loc_from={self.loc_from}, loc_to={self.loc_to}, obs_mask={self.obs_mask}, distance={self.distance})"

    def to_path(self) -> List['CompressedEdge']:
        if self.loc_from is None and self.loc_to is None:
            raise ValueError(f"Both loc_from and loc_to are None")
        elif self.loc_from is None:
            return list(reversed([e.reversed() for e in self.reversed().to_path()]))
        explored = []
        queue: List[Tuple[int, int, 'DetectorNode']] = []
        current_node = self.loc_from
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
            if current_node is None and self.loc_to is None:
                # Found boundary (self.loc_to)
                min_bound_distance, min_bound_obs = _min_edge_to_boundary(boundary_predecessor)
                res = _path_from_predecessors(boundary_predecessor)
                res.append(CompressedEdge(loc_from=boundary_predecessor, loc_to=None,
                                          obs_mask=min_bound_obs, distance=min_bound_distance))
                break
            elif current_node is self.loc_to and current_node is not None:
                # Found self.loc_to
                assert distance == self.distance
                res = _path_from_predecessors(current_node)
                break
            for i in range(len(current_node.neighbors)):
                v = current_node.neighbors_with_boundary[i]
                new_distance = distance + current_node.distances[i]
                if v is not None and v.distance_from_search_source is None:
                    explored.append(v)
                if v is None:
                    if self.loc_to is not None:
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
        src2 = self.loc_to.loc if self.loc_to is not None else None
        raise ValueError(f"No path found from {self.loc_from.loc} to {src2}")

    def draw(self, screen: 'pygame.Surface', scale: float, rgb: Tuple[int, int, int], width: int = 4) -> None:
        """Draws the compressed edge, provided DetectorNode.loc is of type `complex'"""
        import pygame
        if self.loc_from is not None and self.loc_to is not None:
            pygame.draw.line(
                screen,
                rgb,
                (int(self.loc_from.loc.real * scale), int(self.loc_from.loc.imag * scale)),
                (int(self.loc_to.loc.real * scale), int(self.loc_to.loc.imag * scale)),
                width=width,
            )

    def draw_path(self, screen: 'pygame.Surface', scale: float, rgb: Tuple[int, int, int], width: int = 4) -> None:
        for edge in self.to_path():
            edge.draw(screen=screen, scale=scale, rgb=rgb, width=width)
