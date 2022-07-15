import heapq
from typing import (
    TypeVar,
    List,
    Generic,
    Optional,
    TYPE_CHECKING
)

import networkx as nx

from slowmatch.graph import Graph, LocationData, graph_from_networkx
from slowmatch.graph_fill_region import GraphFillRegion
from slowmatch.logger import Logger
from slowmatch.varying import Varying
from slowmatch.events import (BlossomImplodeEvent, RegionHitBoundaryEvent, RegionHitRegionEvent, TentativeEvent,
                              TentativeNeighborInteractionEvent, MwpmEvent, TentativeRegionShrinkEvent)
from slowmatch.region_path import RegionPath
from slowmatch.compressed_edge import CompressedEdge

if TYPE_CHECKING:
    pass

TLocation = TypeVar('TLocation')


def flooder_from_networkx(graph: nx.Graph) -> 'GraphFlooder':
    new_graph = graph_from_networkx(graph)
    return GraphFlooder(graph=new_graph)


class GraphFlooder(Generic[TLocation]):
    def __init__(
        self,
        graph: Graph,
        enable_logger: bool = False
    ):
        self.graph = graph
        self.enable_logger = enable_logger
        self.time = 0
        self._next_event_id = 0
        self._next_region_id = 0
        self._sorted_schedule: List[TentativeEvent] = []
        self.logger = Logger(enabled=self.enable_logger)

    def reset(self):
        self.time = 0
        self._next_event_id = 0
        self._next_region_id = 0
        self._sorted_schedule: List[TentativeEvent] = []

    def create_region(self, location: TLocation) -> GraphFillRegion:
        if self.logger.enabled:
            self.logger.log_region_created()
        k = self._next_region_id
        self._next_region_id += 1
        location_data = self.graph.nodes[location]
        new_region = GraphFillRegion(id=k, source=location_data)
        self._do_region_arriving_at_empty_location(region=new_region, location_data=location_data)
        return new_region

    def next_event(self, max_time: float = float('inf')) -> Optional[MwpmEvent]:
        # Process events until interaction is needed or no more work can be done.
        while self._sorted_schedule and self._sorted_schedule[0].time <= max_time:
            # Get next valid event.
            tentative_event: TentativeEvent = heapq.heappop(self._sorted_schedule)
            if tentative_event.is_invalidated:
                continue
            else:
                tentative_event.invalidate()

            # Apply transformations based on the event.
            self.time = tentative_event.time
            if isinstance(tentative_event, TentativeNeighborInteractionEvent):
                if tentative_event.location_data_2 is not None:
                    mwpm_event = self._do_neighbor_interaction(event=tentative_event)
                else:
                    mwpm_event = self._do_region_hit_boundary_interaction(event=tentative_event)
            elif isinstance(tentative_event, TentativeRegionShrinkEvent):
                mwpm_event = self._do_region_shrinking(event=tentative_event)
            else:
                raise NotImplementedError(f'Unrecognized event: {tentative_event}')

            # If the event requires an update to the MWPM state, return it.
            if mwpm_event is not None:
                return mwpm_event

        # Nothing more to do right now.
        return None

    def has_valid_events_queued(self) -> bool:
        return any(not e.is_invalidated for e in self._sorted_schedule)

    def set_region_growth(self, region: GraphFillRegion, *, new_growth: int):
        region.radius = region.radius.then_slope_at(time_of_change=self.time, new_slope=new_growth)
        self._reschedule_events_for_region(region)

    def create_blossom(self, contained_regions: 'RegionPath') -> GraphFillRegion:
        if self.logger.enabled:
            blossom_depth = 1 + max(r.region.blossom_depth() for r in contained_regions)
            self.logger.log_blossom_created(len(contained_regions), blossom_depth)
        k = self._next_region_id
        self._next_region_id += 1
        blossom_region = GraphFillRegion(id=k)
        blossom_region.radius = Varying(base_time=self.time, slope=1)
        blossom_region.blossom_children = RegionPath()

        for blossom_edge in contained_regions:
            blossom_region.blossom_children.edges.append(blossom_edge)
            blossom_edge.region.radius = blossom_edge.region.radius.then_slope_at(time_of_change=self.time, new_slope=0)
            blossom_edge.region.blossom_parent = blossom_region
            blossom_edge.region.alt_tree_node = None

        self._reschedule_events_for_region(blossom_region)

        # Rescheduling the blossom region fixed location schedules, but not
        # child region schedules. Fix them now.
        for child in blossom_region.blossom_children:
            child.region.invalidate_involved_schedule_items()

        return blossom_region

    def _schedule_tentative_neighbor_interaction_event(
        self, location_data_1: LocationData, schedule_list_index_1: int,
            location_data_2: Optional[LocationData], schedule_list_index_2: Optional[int],
            time: float
    ):
        tentative_event = TentativeNeighborInteractionEvent(
            time=time, event_id=self._next_event_id,
            location_data_1=location_data_1,
            schedule_list_index_1=schedule_list_index_1,
            location_data_2=location_data_2,
            schedule_list_index_2=schedule_list_index_2
        )
        self._next_event_id += 1
        location_data_1.schedule_list[schedule_list_index_1] = tentative_event
        if location_data_2 is not None:
            location_data_2.schedule_list[schedule_list_index_2] = tentative_event
        heapq.heappush(self._sorted_schedule, tentative_event)
        assert time >= self.time

    def _schedule_tentative_shrink_event(
            self, region: GraphFillRegion
    ):
        if not region.shell_area:
            time = region.radius.zero_intercept()  # Blossom implosion
        else:
            time = region.shell_area[-1].local_radius().zero_intercept()  # Leave event, or degenerate implosion
        tentative_event = TentativeRegionShrinkEvent(time=time, event_id=self._next_event_id, region=region)
        self._next_event_id += 1
        region.shrink_event = tentative_event
        heapq.heappush(self._sorted_schedule, tentative_event)
        assert time >= self.time

    def _reschedule_events_for_region(self, region: 'GraphFillRegion'):
        region.invalidate_involved_schedule_items()
        if self.logger.enabled:
            self.logger.log_area_set(region.total_area_size())
        if region.radius.slope < 0:
            self._schedule_tentative_shrink_event(region)
            for location_data in region.iter_total_area():
                location_data.invalidate_involved_schedule_items()
        else:
            for location_data in region.iter_total_area():
                self.reschedule_events_at_location(location_data=location_data)

    def reschedule_events_at_location(self, *, location_data: 'LocationData'):
        location_data.invalidate_involved_schedule_items()

        rad1 = location_data.local_radius()
        for i in range(location_data.num_neighbors):
            distance = location_data.distances[i]
            neighbor_location_data = location_data.neighbors[i]
            if neighbor_location_data is None:
                dis_to_boundary = distance - rad1
                if dis_to_boundary.slope < 0:
                    self._schedule_tentative_neighbor_interaction_event(
                        location_data_1=location_data,
                        schedule_list_index_1=i,
                        location_data_2=None,
                        schedule_list_index_2=None,
                        time=dis_to_boundary.zero_intercept()
                    )
                continue
            if location_data.has_same_owner_as(neighbor_location_data):
                continue
            rad2 = neighbor_location_data.local_radius()
            rad3 = rad1 + rad2 - distance   # Collision time
            if rad3.slope <= 0:
                continue
            j = location_data.neighbor_index[i]
            self._schedule_tentative_neighbor_interaction_event(
                location_data_1=location_data,
                schedule_list_index_1=i,
                location_data_2=neighbor_location_data,
                schedule_list_index_2=j,
                time=rad3.zero_intercept()
            )

    def _do_region_arriving_at_empty_location(
        self, *, region: GraphFillRegion, location_data: LocationData,
            from_location_data: Optional[LocationData] = None, neighbor_index: Optional[int] = None
    ):

        assert region.radius.slope > 0

        assert location_data.reached_from_source is None
        if from_location_data is not None:
            edge_obs = from_location_data.observables[neighbor_index]
            location_data.observables_crossed = from_location_data.observables_crossed ^ edge_obs
            location_data.reached_from_source = from_location_data.reached_from_source
            edge_dist = from_location_data.distances[neighbor_index]
            location_data.distance_from_source = from_location_data.distance_from_source + edge_dist
        else:
            location_data.reached_from_source = location_data
            location_data.distance_from_source = 0
        location_data.region_that_arrived = region
        region.shell_area.append(location_data)
        self.reschedule_events_at_location(location_data=location_data)

    def _do_region_shrinking(self, *, event: TentativeRegionShrinkEvent) -> Optional[MwpmEvent]:
        if not event.region.shell_area:
            return self._do_blossom_implosion(event.region)
        elif event.region.shell_area[-1] is event.region.source:
            return self._do_degenerate_implosion(event.region)
        else:
            location_data = event.region.shell_area.pop()
            assert event.region.radius.slope < 0
            assert location_data.reached_from_source is not None
            assert location_data.region_that_arrived is event.region
            location_data.region_that_arrived = None
            location_data.reached_from_source = None
            location_data.distance_from_source = None
            location_data.observables_crossed = 0
            self.reschedule_events_at_location(location_data=location_data)
            self._schedule_tentative_shrink_event(region=event.region)

    def _do_neighbor_interaction(
        self, *, event: TentativeNeighborInteractionEvent
    ) -> Optional[MwpmEvent]:
        loc_data_1 = event.location_data_1
        loc_data_2 = event.location_data_2
        assert not loc_data_1.is_empty() or not loc_data_2.is_empty()

        # One region spreading into an empty location?
        idx1, idx2 = event.schedule_list_index_1, event.schedule_list_index_2
        if not loc_data_2.is_empty() and loc_data_1.is_empty():
            loc_data_1, loc_data_2 = loc_data_2, loc_data_1
            idx1, idx2 = idx2, idx1

        region1 = loc_data_1.top_region()

        if loc_data_2.is_empty():
            self._do_region_arriving_at_empty_location(
                region=region1,
                location_data=loc_data_2,
                from_location_data=loc_data_1,
                neighbor_index=idx1
            )
            return None

        # Two regions colliding.
        region2 = loc_data_2.top_region()
        coll_obs_mask = (loc_data_1.observables_crossed ^ loc_data_2.observables_crossed
                         ^ loc_data_1.observables[event.schedule_list_index_1])
        coll_dist = (loc_data_1.distance_from_source + loc_data_2.distance_from_source
                     + loc_data_1.distances[event.schedule_list_index_1])
        return RegionHitRegionEvent(region1=region1, region2=region2, time=self.time,
                                    edge=CompressedEdge(
                                        source1=loc_data_1.reached_from_source,
                                        source2=loc_data_2.reached_from_source,
                                        obs_mask=coll_obs_mask, distance=coll_dist)
                                    )

    def _do_region_hit_boundary_interaction(
            self, event: TentativeNeighborInteractionEvent
    ) -> MwpmEvent:
        loc_data = event.location_data_1
        neighbour_index = event.schedule_list_index_1
        region = loc_data.top_region()
        obs_crossed_on_path = loc_data.observables_crossed ^ loc_data.observables[neighbour_index]
        distance_along_path = loc_data.distance_from_source + loc_data.distances[neighbour_index]
        neighbour_loc = loc_data.neighbors_with_boundary[neighbour_index]
        return RegionHitBoundaryEvent(
            time=self.time,
            region=region,
            edge=CompressedEdge(
                source1=loc_data.reached_from_source,
                source2=neighbour_loc,
                obs_mask=obs_crossed_on_path,
                distance=distance_along_path
            )
        )

    def _do_degenerate_implosion(self, region: 'GraphFillRegion') -> Optional[MwpmEvent]:
        assert region.radius(self.time) == 0
        assert region.radius.slope < 0
        alt_parent = region.alt_tree_node.parent
        alt_child = region.alt_tree_node.inner_outer_edge

        parent_region = alt_parent.node.outer_region
        child_region = region.alt_tree_node.outer_region

        return RegionHitRegionEvent(
            time=self.time,
            region1=parent_region,
            region2=child_region,
            edge=alt_parent.edge.reversed() & alt_child
        )

    def _do_blossom_implosion(self, region: 'GraphFillRegion') -> Optional[MwpmEvent]:
        assert region.radius(self.time) == 0
        assert region.radius.slope < 0
        if self.logger.enabled:
            bsize = len(region.blossom_children) if region.blossom_children else 0
            self.logger.log_blossom_implosion(bsize, region.blossom_depth())
        assert region.blossom_children

        for child in region.blossom_children:
            child.region.blossom_parent = None

        in_parent_region = region.alt_tree_node.parent.edge.source1.top_region()
        in_child_region = region.alt_tree_node.inner_outer_edge.source1.top_region()
        return BlossomImplodeEvent(
            time=self.time,
            blossom_region=region,
            in_parent_region=in_parent_region,
            in_child_region=in_child_region
        )
