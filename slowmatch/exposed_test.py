import random
import time

import networkx
import pydot
from typing import List, Tuple

from slowmatch import embedded_min_weight_match, embedded_min_weight_match_extended
from slowmatch.graph_flooder import GraphFlooder


def test_embedded_min_weight_match():
    g = pydot.graph_from_dot_data(
        r"""
    graph {
        // Define boundary nodes.
        LEFT [boundary];
        RIGHT [boundary];
        // Define detection event nodes.
        b [detection];
        e [detection];
        // Other nodes are implicitly defined by edge declarations.
        LEFT -- a [weight=1];
        a -- b;  // Default weight is 1.
        b -- c -- d [weight=3]; // Multi-edge declarations work.
        d -- e [weight=2];
        e -- RIGHT [weight=2];
    }
        """
    )
    graph = networkx.nx_pydot.from_pydot(g[0])
    result = embedded_min_weight_match(graph)
    assert set(result) == {('b', 'LEFT'), ('e', 'RIGHT')}


def rep_neighbors(pos: complex) -> List[Tuple[int, complex]]:
    return [
        (141, pos - 1j - 1),
        (141, pos + 1j + 1),
        (103, pos - 1),
        (103, pos + 1),
        (100, pos - 1j),
        (100, pos + 1j),
    ]


def generate_rep_code_detection_events(width: int, height: int, error_rate: float):
    detection_events = set()

    def flip_detection_event(loc):
        if not 0 < loc.real < width or not 0 < loc.imag < height:
            return
        if loc in detection_events:
            detection_events.remove(loc)
        else:
            detection_events.add(loc)

    for i in range(-1, width + 1):
        for j in range(-1, height + 1):
            c = i + j * 1j
            for _, other in rep_neighbors(c):
                if (c.real, c.imag) < (other.real, other.imag) and random.random() < error_rate:
                    flip_detection_event(c)
                    flip_detection_event(other)

    return list(detection_events)


def test_rep_code_graph():
    width = 100
    height = 100

    def complex_grid_boundary(pos: complex) -> bool:
        return not 0 < pos.real < width or not 0 < pos.imag < height

    detection_events = generate_rep_code_detection_events(width, height, 0.01)
    embedded_min_weight_match_extended(
        flooder=GraphFlooder(neighbors=rep_neighbors,
                             boundary=complex_grid_boundary),
        detection_events=detection_events
    )
