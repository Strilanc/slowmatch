import random

import networkx
import pydot
from typing import List, Tuple
import stim
import numpy as np
import pytest
import warnings
import pymatching
import networkx as nx

from slowmatch.exposed import (Matching, graph_from_neighbors_and_boundary, embedded_min_weight_match,
                               detector_error_model_to_nx_graph, discretize_weights, set_bits)
from slowmatch.compressed_edge import CompressedEdge
from slowmatch.graph import DetectorNode


def nx_graph_to_pymatching(g: nx.Graph) -> pymatching.Matching:
    for _, _, d in g.edges(data=True):
        d["fault_ids"] = set_bits(d["observables"])
    max_node_id = max(g.nodes())
    max_fault_id = max(max(d["fault_ids"], default=0) for _, _, d in g.edges(data=True))
    g.add_edge(max_node_id, max_node_id + 1, weight=9999999999, fault_ids=max_fault_id)
    for i in range(len(g.nodes)):
        g.nodes[i]["is_boundary"] = g.nodes[i].get("boundary", False)
    return pymatching.Matching(g)


def detector_error_model_to_discretised_pymatching_graph(
        model: stim.DetectorErrorModel, num_buckets: int = 1000
    ) -> pymatching.Matching:
    graph = detector_error_model_to_nx_graph(model)
    graph = discretize_weights(graph, num_buckets)
    return nx_graph_to_pymatching(graph)


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = networkx.nx_pydot.from_pydot(g[0])
    result = embedded_min_weight_match(graph)
    m1, m2 = result.match_edges
    assert m1.loc_from.loc == "b"
    assert m1.loc_to is "LEFT"
    assert m2.loc_from.loc == "e"
    assert m2.loc_to is "RIGHT"


def rep_neighbors(pos: complex) -> List[Tuple[int, int, complex]]:
    return [
        (141, 0, pos - 1j - 1),
        (141, 0, pos + 1j + 1),
        (103, 0, pos - 1),
        (103, 0, pos + 1),
        (100, 0, pos - 1j),
        (100, 0, pos + 1j),
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
            for _, _, other in rep_neighbors(c):
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
    graph = graph_from_neighbors_and_boundary(1 + 1j, rep_neighbors, complex_grid_boundary)
    matching = Matching(model=graph)
    res = matching.decode_from_event_locations(detection_events)


DETECTION_COUNT = 20
SCALE = 20
TIME_SCALE = 100
ERROR_RATE = 0.1
WIDTH = 800
HEIGHT = 600


def complex_grid_neighbors(pos: complex) -> List[Tuple[int, int, complex]]:
    return [
        (141, 0, pos - 1j - 1),
        (141, 0, pos + 1j + 1),
        (100, 0, pos - 1),
        (100, 0, pos + 1),
        (100, 0, pos - 1j),
        (100, 0, pos + 1j),
    ]


def complex_grid_boundary(pos: complex) -> bool:
    return not 0 < pos.real < WIDTH / SCALE or not 0 < pos.imag < HEIGHT / SCALE


def plausible_case(width: int, height: int, p: float = ERROR_RATE):
    kept = set()

    def flip(x):
        if complex_grid_boundary(x):
            return
        if x in kept:
            kept.remove(x)
        else:
            kept.add(x)

    for i in range(-1, width // SCALE + 1):
        for j in range(-1, height // SCALE + 1):
            c = i + j * 1j
            for _, _, other in complex_grid_neighbors(c):
                if (c.real, c.imag) < (other.real, other.imag):
                    if random.random() < p:
                        flip(c)
                        flip(other)

    return list(kept)


def test_complex_grid():
    random.seed(3)
    for i in range(20):
        locs = plausible_case(WIDTH, HEIGHT)
        graph = graph_from_neighbors_and_boundary(1 + 1j, complex_grid_neighbors,
                                                  complex_grid_boundary)
        matching = Matching(model=graph)
        res = matching.decode_from_event_locations(locs)
        match_locs = set()
        for e in res.match_edges:
            match_locs.add(e.loc_from.loc)
            if not complex_grid_boundary(e.loc_to.loc):
                match_locs.add(e.loc_to.loc)
        assert match_locs == set(locs)


@pytest.mark.parametrize("d,noise,seed,num_shots",
                         [(7, 0.01, 0, 20)]
                         )
def test_slowmatch_vs_pymatching(d: int, noise: float, seed: int, num_shots: int):
    circuit = stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        rounds=d,
        distance=d,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise)
    shots = circuit.compile_detector_sampler(seed=seed).sample(num_shots, append_observables=True)

    detector_parts = shots[:, :circuit.num_detectors]

    error_model = circuit.detector_error_model(decompose_errors=True)
    num_buckets = 1000
    matching_graph = Matching(error_model, num_buckets=num_buckets)

    pymatching_graph = detector_error_model_to_discretised_pymatching_graph(error_model, num_buckets=num_buckets)

    num_shots = detector_parts.shape[0]
    num_dets = circuit.num_detectors
    assert detector_parts.shape[1] == num_dets

    for k in range(num_shots):
        expanded_det = np.resize(detector_parts[k], num_dets + 1)
        expanded_det[-1] = 0
        res = matching_graph.decode(expanded_det)
        pm_prediction, pm_weight = pymatching_graph.decode(expanded_det, return_weight=True, num_neighbours=None)
        assert pm_weight == res.total_weight
