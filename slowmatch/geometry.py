import cmath
from typing import Iterable, List, TYPE_CHECKING, Dict

import math
from scipy.spatial import Voronoi
import numpy as np

if TYPE_CHECKING:
    from slowmatch.graph import DetectorNode


def is_left_turn(p1: complex, p2: complex, p3: complex) -> bool:
    """Determines whether the points p1->p2->p3 make a left turn."""
    p2 -= p1
    p3 -= p1
    return (p2.conjugate() * p3).imag <= 0


def graham_scan(points: Iterable[complex]) -> List[complex]:
    """
    Find the convex hull of the points
    """
    points = set(points)
    if len(points) <= 2:
        return list(points)
    center = sum(points) / len(points)
    points = sorted(points, key=lambda x: cmath.phase(x - center))

    stack = []
    for p in points:
        while len(stack) > 1 and is_left_turn(stack[-2], stack[-1], p):
            stack.pop()
        stack.append(p)
    return stack


def get_unit_radius_polygon_around_node(source: 'DetectorNode') -> List[complex]:
    rel_neighbors = [n.loc - source.loc for n in source.neighbors_with_boundary]
    corners = [rel_neighbors[i] / source.neighbor_distances[i] for i in range(len(source.neighbors))]
    return sorted(corners, key=lambda x: math.atan2(x.imag, x.real))


def voronoi_from_points(points: List[complex]) -> Dict[complex, List[complex]]:
    points_array = np.array([[x.real, x.imag] for x in points])
    vor = Voronoi(points_array)
    out = {}
    for i, region_id in enumerate(vor.point_region):
        region = vor.regions[region_id]
        region_coords = [vor.vertices[idx, 0] + vor.vertices[idx, 1] * 1j if idx != -1 else -1 for idx in region]
        out[points[i]] = region_coords if -1 not in region_coords else []
    return out
