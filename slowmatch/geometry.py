from typing import Iterable, List, TYPE_CHECKING, Dict

import math
from scipy.spatial import Voronoi
import numpy as np

if TYPE_CHECKING:
    from slowmatch.graph import LocationData


def ccw(p1: complex, p2: complex, p3: complex):
    """
    Determines whether the points p1->p2->p3 make a left turn.
    Returns positive if they make a left turn, negative if they make
    a right turn, or zero if they are colinear.
    """
    return (p2.real - p1.real) * (p3.imag - p1.imag) - (p2.imag - p1.imag) * (p3.real - p1.real)


def presort_for_graham_scan(points: Iterable[complex]) -> List[complex]:
    points = set(points)
    p0 = min(points, key=lambda x: (x.imag, x.real))
    points.remove(p0)
    points = sorted(points, key=lambda x: -(x - p0).real / abs(x - p0))
    return [p0] + points


def graham_scan(points: Iterable[complex]) -> List[complex]:
    """
    Find the convex hull of the points
    """
    points = presort_for_graham_scan(points)
    stack = []
    for p in points:
        while len(stack) > 1 and ccw(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)
    return stack


def sort_counter_clockwise_around_source(points: Iterable[complex], source: complex) -> List[complex]:
    return sorted(points, key=lambda x: math.atan2((x-source).imag, (x-source).real))


def get_unit_radius_polygon_around_node(source: 'LocationData') -> List[complex]:
    rel_neighbors = [n.loc - source.loc for n in source.neighbors_with_boundary]
    corners = [rel_neighbors[i]/source.distances[i] for i in range(len(source.neighbors))]
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
