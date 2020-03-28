from typing import Tuple, Any, List, Iterable

import networkx

from slowmatch.flooder import Flooder
from slowmatch.graph_flooder import GraphFlooder
from slowmatch.mwpm import Mwpm


def embedded_min_weight_match(graph: networkx.Graph) -> List[Tuple[Any, Any]]:
    def neighbors(node):
        for edge in graph.edges([node]):
            opp = edge[1]
            e = graph.get_edge_data(node, opp)
            weight = e.get('weight', 1)
            assert isinstance(weight, int)
            yield weight, opp

    def boundary(node):
        return graph.nodes[node].get('boundary', False) in [True, None]

    flooder = GraphFlooder(neighbors, boundary)
    detection_events = [n for n in graph.nodes if graph.nodes[n].get('detection', False) in [True, None]]
    return embedded_min_weight_match_extended(flooder, detection_events)


def embedded_min_weight_match_extended(flooder: Flooder,
                                       detection_events: Iterable[Any]) -> List[Tuple[Any, Any]]:
    mwpm = Mwpm(flooder)

    # Add detection events.
    back_map = {}
    for d in detection_events:
        k = flooder.create_region(d)
        mwpm.add_region(k)
        back_map[k] = d

    # Process.
    while True:
        event = flooder.next_event()
        if event is None:
            break
        mwpm.process_event(event)

    # Report matches.
    seen = set()
    result = []
    for v1, v2 in mwpm.match_map.items():
        if v1 in seen or v2 in seen:
            break
        seen.add(v1)
        seen.add(v2)
        # TODO: EXPAND BLOSSOMS
        if v1 in back_map and v2 in back_map:
            result.append((back_map[v1], back_map[v2]))
    for v1, v2 in mwpm.boundary_match_map.items():
        # TODO: EXPAND BLOSSOMS
        if v1 in back_map:
            result.append((back_map[v1], v2))
    return result
