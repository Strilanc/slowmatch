from typing import Optional

import cirq
import pytest
import stim
import numpy as np

from slowmatch.compressed_edge import CompressedEdge, join_edges
from slowmatch.graph import LocationData, Graph
from slowmatch.exposed import Matching


l0 = LocationData(loc=0)
l1 = LocationData(loc=1)
l2 = LocationData(loc=2)
l3 = LocationData(loc=3)
l4 = LocationData(loc=4)
l5 = LocationData(loc=5)


def test_compressed_edge_equality():
    eq = cirq.testing.EqualsTester()
    a = CompressedEdge(loc_from=l0, loc_to=l1, obs_mask=0, distance=1)
    b = CompressedEdge(loc_from=l0, loc_to=l2, obs_mask=0, distance=1)
    c = CompressedEdge(loc_from=l0, loc_to=l1, obs_mask=0, distance=1)
    d = CompressedEdge(loc_from=l0, loc_to=l1, obs_mask=1, distance=1)
    e = CompressedEdge(loc_from=l0, loc_to=l1, obs_mask=1, distance=2)
    eq.add_equality_group(a, c)
    eq.add_equality_group(b)
    eq.add_equality_group(d)
    eq.add_equality_group(e)


def test_reversed_compressed_edge():
    a = CompressedEdge(
        loc_from=l0,
        loc_to=l1,
        obs_mask=0,
        distance=1
    )
    b = CompressedEdge(
        loc_from=l1,
        loc_to=l0,
        obs_mask=0,
        distance=1
    )
    assert a.reversed() == b


def test_merge_compressed_edge():
    a = CompressedEdge(
        loc_from=l0,
        loc_to=l1,
        obs_mask=1,
        distance=1
    )
    b = CompressedEdge(
        loc_from=l1,
        loc_to=l2,
        obs_mask=2,
        distance=10
    )
    c = CompressedEdge(
        loc_from=l2,
        loc_to=l3,
        obs_mask=4,
        distance=110
    )
    d = CompressedEdge(
        loc_from=l3,
        loc_to=l4,
        obs_mask=5,
        distance=1000
    )
    assert a.merged_with(b) == CompressedEdge(loc_from=l0, loc_to=l2, obs_mask=3, distance=11)
    assert a & b & c == CompressedEdge(loc_from=l0, loc_to=l3, obs_mask=7, distance=121)
    assert b & c & d == CompressedEdge(loc_from=l1, loc_to=l4, obs_mask=3, distance=1120)


def test_merging_incompatible_compressed_edges_raises_value_error():
    a = CompressedEdge(
        loc_from=l0,
        loc_to=l1,
        obs_mask=1,
        distance=1
    )
    b = CompressedEdge(
        loc_from=l2,
        loc_to=l3,
        obs_mask=2,
        distance=10
    )
    with pytest.raises(ValueError):
        a.merged_with(b)


def compressed_edge_generator(graph: Graph):
    def make_edge(i: Optional[int], j: Optional[int], o: int, w: int):
        src1 = graph.nodes[i] if i is not None else None
        src2 = graph.nodes[j] if j is not None else None
        return CompressedEdge(loc_from=src1, loc_to=src2, obs_mask=o, distance=w)
    return make_edge


def test_to_path():
    g = Graph()
    g.add_edge(0, 1, 1, 0)
    g.add_edge(0, 2, 1, 0)
    g.add_edge(1, 3, 1, 0)
    g.add_edge(2, 4, 2, 0)
    g.add_edge(3, 5, 3, 0)
    g.add_edge(4, 6, 1, 0)
    g.add_edge(5, 7, 4, 0)
    g.add_edge(6, 7, 4, 0)

    ce = compressed_edge_generator(g)

    e = ce(0, 7, 0, 8)

    assert e.to_path() == [ce(0, 2, 0, 1), ce(2, 4, 0, 2), ce(4, 6, 0, 1), ce(6, 7, 0, 4)]


def test_path_to_boundary():
    g = Graph()
    g.add_boundary_edge(0, 2, 1)
    g.add_edge(0, 1, 1, 0)
    g.add_edge(1, 2, 3, 5)
    g.add_edge(2, 3, 0, 0)
    g.add_boundary_edge(3, weight=5, observables=5)

    ce = compressed_edge_generator(g)

    assert ce(0, None, 1, 2).to_path() == [ce(0, None, 1, 2)]
    assert  ce(2, None, 5, 5).to_path() == [ce(2, 3, 0, 0), ce(3, None, 5, 5)]
    assert ce(None, 2, 5, 5).to_path() == [ce(None, 3, 5, 5), ce(3, 2, 0, 0)]


@pytest.mark.parametrize(
    "d,noise,num_shots,seed",
    [
        (5, 0.005, 20, 0),
        (7, 0.01, 20, 0)
    ]
)
def test_edge_to_path_for_matching(d, noise, num_shots, seed):
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
    num_shots = detector_parts.shape[0]
    num_dets = circuit.num_detectors
    assert detector_parts.shape[1] == num_dets

    for k in range(num_shots):
        expanded_det = np.resize(detector_parts[k], num_dets + 1)
        expanded_det[-1] = 0
        res = matching_graph.decode(expanded_det)
        for e in res.match_edges:
            p = e.to_path()
            assert sum(x.distance for x in p) == e.distance
            m = join_edges(p)
            assert m.loc_from == e.loc_from
            assert m.loc_to == e.loc_to
