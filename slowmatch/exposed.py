import dataclasses
import math
from typing import List, Callable, Union, Set, Any

import networkx as nx
import numpy as np
import stim

from slowmatch.compressed_edge import CompressedEdge
from slowmatch.graph import Graph, graph_from_networkx
from slowmatch.graph_flooder import GraphFlooder
from slowmatch.mwpm import Mwpm


def graph_from_neighbors_and_boundary(seed: complex, neighbors_func: Callable[[Any], List[Any]], is_boundary_func: Callable[[Any], bool]) -> Graph:
    g = Graph()
    edges = set()
    seen = set()
    q = [seed]
    while len(q) > 0:
        u = q.pop()
        seen.add(u)
        for w, obs, neighbor in neighbors_func(u):
            if is_boundary_func(neighbor):
                g.add_boundary_edge(u, weight=w, observables=obs, boundary_node=neighbor)
            elif (u, neighbor) not in edges:
                g.add_edge(u, neighbor, weight=w, observables=obs)
                edges.add((u, neighbor))
                edges.add((neighbor, u))
                if neighbor not in seen:
                    q.append(neighbor)
    return g


def embedded_min_weight_match(graph: nx.Graph):
    detection_events = [n for n in graph.nodes if graph.nodes[n].get('detection', False) in [True, None]]
    matching = Matching(graph)
    return matching.decode_from_event_locations(detection_events=detection_events)


@dataclasses.dataclass
class MatchingResult:
    match_edges: List['CompressedEdge']
    total_weight: int
    predicted_observables: np.ndarray


class Matching:
    def __init__(
            self,
            model: Union[stim.DetectorErrorModel, nx.Graph, Graph],
            num_buckets: int = 1000,
            enable_logger: bool = False
    ):
        if isinstance(model, stim.DetectorErrorModel):
            self.model: stim.DetectorErrorModel = model
            self.graph = graph_from_networkx(discretize_weights(detector_error_model_to_nx_graph(model),
                                                                num_buckets=num_buckets))
            self.num_observables = self.model.num_observables
        elif isinstance(model, nx.Graph):
            self.graph = graph_from_networkx(discretize_weights(model, num_buckets=num_buckets))
            self.num_observables = self.graph.num_observables
        elif isinstance(model, Graph):
            self.graph = model
            self.num_observables = self.graph.num_observables
        else:
            raise ValueError(f"model type {type(model)} not recognised")
        self.flooder = GraphFlooder(self.graph, enable_logger=enable_logger)
        self.mwpm = Mwpm(self.flooder)

    def decode_from_event_locations(self, detection_events: List[Union[int, complex]]) -> MatchingResult:
        # Add detection events
        for d in detection_events:
            self.mwpm.add_detection_event(d)

        # Process.
        while True:
            event = self.flooder.next_event()
            if event is None:
                break
            self.mwpm.process_event(event)

        # Shatter blossoms to extract matching and cleanup Mwpm object for next cycle
        match_edges, total_weight, obs_mask = self.mwpm.extract_matching_and_reset_graph()
        obs_array = int_to_binary_array(obs_mask, self.num_observables)
        self.mwpm.reset()
        return MatchingResult(
            match_edges=match_edges,
            total_weight=total_weight,
            predicted_observables=obs_array
        )

    def decode(self, syndrome) -> MatchingResult:
        detection_events = syndrome.nonzero()[0]
        return self.decode_from_event_locations(detection_events=detection_events)


def int_to_binary_array(n: int, num_bits: int) -> np.ndarray:
    obs_list = [int(i) for i in bin(n)[:1:-1]]
    obs_list += [0] * (num_bits - len(obs_list))
    return np.array(obs_list, dtype=np.uint8)


def set_bits(n: int) -> Set[int]:
    return {i for i, c in enumerate(bin(n)[:1:-1]) if c == '1'}


def discretize_weights(g: nx.Graph, num_buckets) -> nx.Graph:
    max_weight = max(float(e[2].get("weight", 1)) for e in g.edges(data=True))
    min_weight = min(float(e[2].get("weight", 1)) for e in g.edges(data=True))
    if min_weight < 0:
        raise NotImplementedError("Negative weights not yet supported")
    for u, v, d in g.edges(data=True):
        d["weight"] = int(float(d.get("weight", 1)) * num_buckets / max_weight)
    return g


def detector_error_model_to_nx_graph(model: stim.DetectorErrorModel) -> nx.Graph:
    """Convert a stim error model into a NetworkX graph.
    From: https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb"""

    g = nx.Graph()
    boundary_node = model.num_detectors
    g.add_node(boundary_node, boundary=True, coords=[-1, -1, -1])

    def handle_error(p: float, dets: List[int], frame_changes: int):
        if p == 0:
            return
        if len(dets) == 0:
            # No symptoms for this error.
            # Code probably has distance 1.
            # Accept it and keep going, though of course decoding will probably perform terribly.
            return
        if len(dets) == 1:
            dets = [dets[0], boundary_node]
        if len(dets) > 2:
            raise NotImplementedError(
                f"Error with more than 2 symptoms can't become an edge or boundary edge: {dets!r}.")
        if g.has_edge(*dets):
            edge_data = g.get_edge_data(*dets)
            old_p = edge_data["error_probability"]
            old_frame_changes = edge_data["observables"]
            # If frame changes differ, the code has distance 2; just keep whichever was first.
            if old_frame_changes == frame_changes:
                p = p * (1 - old_p) + old_p * (1 - p)
                g.remove_edge(*dets)
        g.add_edge(*dets, weight=math.log((1 - p) / p), observables=frame_changes, error_probability=p)

    def handle_detector_coords(detector: int, coords: np.ndarray):
        g.add_node(detector, coords=coords)

    eval_model(model, handle_error, handle_detector_coords)

    return g


def eval_model(
        model: stim.DetectorErrorModel,
        handle_error: Callable[[float, List[int], int], None],
        handle_detector_coords: Callable[[int, np.ndarray], None]):
    """Interprets the error model instructions, taking care of loops and shifts.
    Adapted from: https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb

    Makes callbacks as error mechanisms are declared, and also when detector
    coordinate data is declared.
    """
    det_offset = 0
    coords_offset = np.zeros(100, dtype=np.float64)

    def _helper(m: stim.DetectorErrorModel, reps: int):
        nonlocal det_offset
        nonlocal coords_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _helper(instruction.body_copy(), instruction.repeat_count)
                elif isinstance(instruction, stim.DemInstruction):
                    if instruction.type == "error":
                        dets: List[int] = []
                        frames: int = 0
                        t: stim.DemTarget
                        p = instruction.args_copy()[0]
                        for t in instruction.targets_copy():
                            if t.is_relative_detector_id():
                                dets.append(t.val + det_offset)
                            elif t.is_logical_observable_id():
                                frames ^= 2**t.val
                            elif t.is_separator():
                                # Treat each component of a decomposed error as an independent error.
                                # (Ideally we could configure some sort of correlated analysis; oh well.)
                                handle_error(p, dets, frames)
                                frames = 0
                                dets = []
                        # Handle last component.
                        handle_error(p, dets, frames)
                    elif instruction.type == "shift_detectors":
                        det_offset += instruction.targets_copy()[0]
                        a = np.array(instruction.args_copy())
                        coords_offset[:len(a)] += a
                    elif instruction.type == "detector":
                        a = np.array(instruction.args_copy())
                        for t in instruction.targets_copy():
                            handle_detector_coords(t.val + det_offset, a + coords_offset[:len(a)])
                    elif instruction.type == "logical_observable":
                        pass
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
    _helper(model, 1)
