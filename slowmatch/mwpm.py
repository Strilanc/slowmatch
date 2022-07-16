import math
from typing import TypeVar, List, Tuple, TYPE_CHECKING, Iterator

from slowmatch.alternating_tree import AltTreeNode, AltTreeEdge
from slowmatch.graph_flooder import (GraphFlooder)
from slowmatch.graph import DetectorNode
from slowmatch.graph_fill_region import Match, GraphFillRegion
from slowmatch.events import (MwpmEvent, RegionHitRegionEvent,
                              BlossomImplodeEvent, RegionHitBoundaryEvent)
from slowmatch.region_path import RegionPath, RegionEdge
from slowmatch.compressed_edge import CompressedEdge


if TYPE_CHECKING:
    import pygame

TLocation = TypeVar('TLocation')


class Mwpm:
    """The internal state of an embedded minimum weight perfect matching algorithm."""

    def __init__(self, flooder: GraphFlooder):
        self.fill_system: GraphFlooder = flooder
        self.detection_events: List[DetectorNode] = []
        self.match_edges: List['CompressedEdge'] = []

    def add_region(self, region: 'GraphFillRegion'):
        outer_node = AltTreeNode(inner_region=None, outer_region=region)
        region.alt_tree_node = outer_node

    def add_detection_event(self, node_id: TLocation):
        location_data = self.fill_system.graph.nodes[node_id]
        region = self.fill_system.create_region(node_id)
        self.add_region(region)
        self.detection_events.append(location_data)

    def process_event(self, event: MwpmEvent):
        if isinstance(event, BlossomImplodeEvent):
            self.handle_blossom_imploding(event)
        elif isinstance(event, RegionHitRegionEvent):
            if event.region1.matched_to_region() or event.region2.matched_to_region():
                self.handle_tree_hitting_match(event)
            elif (
                    event.region1.is_matched_to_boundary() or event.region2.is_matched_to_boundary()
            ):
                self.handle_tree_hitting_boundary_match(event)
            elif event.region1.alt_tree_node.in_same_tree_as(event.region2.alt_tree_node):
                self.handle_tree_hitting_self(event)
            else:
                self.handle_tree_hitting_other_tree(event)
        elif isinstance(event, RegionHitBoundaryEvent):
            self.handle_tree_hitting_boundary(event)
        else:
            raise NotImplementedError(f'Unrecognized event type "{type(event)}": {event!r}')

    def handle_blossom_imploding(self, event: BlossomImplodeEvent):
        blossom_io_node = event.blossom_region.alt_tree_node
        assert blossom_io_node.inner_region is event.blossom_region
        event.blossom_region.alt_tree_node = None

        # Find the inner regions touching the outer parent and child.
        in_parent_region = event.in_parent_region
        in_child_region = event.in_child_region

        # Find even and odd length paths from parent to child.
        inner_regions = event.blossom_region.blossom_children
        odds, matches = inner_regions.split_between_regions(start_region=in_parent_region, end_region=in_child_region)

        # The even length path becomes matches.
        # Set region growth to zero for newly matched regions to ensure they are rescheduled
        for match_region in matches.pairs_matched():
            self.fill_system.set_region_growth(match_region, new_growth=0)
            self.fill_system.set_region_growth(match_region.match.region, new_growth=0)

        # The odd length path is inserted into the alternating tree.
        ancestor = blossom_io_node.parent.node
        ancestor.children = [e for e in ancestor.children if e.node is not blossom_io_node]
        cur_alt_tree_node = ancestor
        child_edge = blossom_io_node.parent.edge.reversed()
        for k in range(0, len(odds) - 1, 2):
            cur_alt_tree_node = cur_alt_tree_node.make_child_inner_outer(
                inner_graph_fill_region=odds[k].region,
                outer_graph_fill_region=odds[k + 1].region,
                inner_outer_edge=odds[k].edge,
                child_edge=child_edge
            )
            child_edge = odds[k + 1].edge
            self.fill_system.set_region_growth(cur_alt_tree_node.inner_region, new_growth=-1)
            self.fill_system.set_region_growth(cur_alt_tree_node.outer_region, new_growth=+1)

        blossom_io_node.inner_region = odds[-1].region
        odds[-1].region.alt_tree_node = blossom_io_node
        blossom_io_node.parent = None
        cur_alt_tree_node.add_child(
            child=AltTreeEdge(
                node=blossom_io_node,
                edge=child_edge
            )
        )
        self.fill_system.set_region_growth(blossom_io_node.inner_region, new_growth=-1)

    def handle_tree_hitting_match(self, event: RegionHitRegionEvent):
        """An outer node from an alternating tree hit a matched node from a match.

        This breaks the match apart, and adds its two nodes into the tree at the point of contact.

        Args:
            event: Event data including the two regions that are colliding.
        """
        if event.region2.match is not None:
            node = event.region1.alt_tree_node
            match = event.region2
            child_edge = event.edge
        else:
            node = event.region2.alt_tree_node
            match = event.region1
            child_edge = event.edge.reversed()
        other_match = match.match
        node.make_child_inner_outer(
            inner_graph_fill_region=match,
            outer_graph_fill_region=other_match.region,
            inner_outer_edge=match.match.edge,
            child_edge=child_edge
        )
        match.match = None
        other_match.region.match = None
        self.fill_system.set_region_growth(match, new_growth=-1)
        self.fill_system.set_region_growth(other_match.region, new_growth=+1)

    def _shatter_descendants_into_matches_and_freeze(self, node: AltTreeNode):
        for matched_region in node.shatter_into_matches():
            self.fill_system.set_region_growth(matched_region, new_growth=0)
            self.fill_system.set_region_growth(matched_region.match.region, new_growth=0)
        node.children = []

    def handle_tree_hitting_boundary(
            self, event: RegionHitBoundaryEvent
    ):
        """
        An outer node from an alternating tree hit a boundary, shattering the node's tree, and
        matching the node to the boundary
        """
        # Match to boundary.
        node = event.region.alt_tree_node
        assert node.outer_region is event.region
        event.region.match = Match(
            region=None,
            edge=event.edge
        )
        event.region.alt_tree_node = None

        # Shatter the alternating tree into matches.
        self.fill_system.set_region_growth(node.outer_region, new_growth=0)
        node.become_root()
        self._shatter_descendants_into_matches_and_freeze(node)

    def handle_tree_hitting_boundary_match(
            self, event: RegionHitRegionEvent
    ):
        incoming = event.region1
        squished = event.region2
        match_edge = event.edge
        if incoming.is_matched_to_boundary():
            incoming, squished = squished, incoming
            match_edge = match_edge.reversed()
        assert not incoming.is_matched_to_boundary()
        node = incoming.alt_tree_node
        incoming.add_match(match=squished, edge=match_edge)
        # Shatter the alternating tree into matches.
        self.fill_system.set_region_growth(node.outer_region, new_growth=0)
        node.become_root()
        self._shatter_descendants_into_matches_and_freeze(node)

    def handle_tree_hitting_self(self, event: RegionHitRegionEvent):
        """Two outer nodes from an alternating tree have hit each other.

        This creates a cycle in the tree, requiring the creation of a blossom encompassing the two
        outer nodes and any nodes on the existing path between them.

        Args:
            event: Event data including the two regions that are colliding.
        """
        region1 = event.region1.alt_tree_node
        region2 = event.region2.alt_tree_node

        # Remove the path from the two nodes to their common ancestor from the tree.
        common_ancestor = region1.most_recent_common_ancestor(region2)
        p1 = region1.prune_upward_path_stopping_before(common_ancestor)
        p2 = region2.prune_upward_path_stopping_before(common_ancestor)
        common_ancestor_children = common_ancestor.children
        for child in common_ancestor_children:
            child.node.parent = None
        common_ancestor.children = []

        # Determine what to add back into the tree.
        orphans: List[AltTreeEdge] = [
            *common_ancestor_children,
            *p1.orphans,
            *p2.orphans,
        ]
        region1_to_region2 = RegionEdge(
                region=event.region1,
                edge=event.edge
            )
        common_ancestor_edge = RegionEdge(
                region=common_ancestor.outer_region,
                edge=None
            )
        region1_to_ancestor_via_region2 = RegionPath(
            edges=[region1_to_region2] + p2.pruned_path_regions.edges + [common_ancestor_edge]
        )

        blossom_region_cycle = p1.pruned_path_regions + region1_to_ancestor_via_region2.reversed()[:-1]
        blossom = self.fill_system.create_blossom(blossom_region_cycle)

        # Update the tree structure.
        common_ancestor.outer_region = blossom
        blossom.alt_tree_node = common_ancestor
        for orphan in orphans:
            common_ancestor.add_child(child=orphan)

    def handle_tree_hitting_other_tree(self, event: RegionHitRegionEvent):
        """Two outer nodes, from different alternating trees, have hit each other.

        This creates an augmenting path between the roots of the two trees, passing through the
        collision. After applying this augmenting path, both trees dissolve into matches.

        Args:
            event: Event data including the two regions that are colliding.
        """
        region1 = event.region1.alt_tree_node
        region2 = event.region2.alt_tree_node

        assert not region1.in_same_tree_as(region2)

        region1.become_root()
        region2.become_root()

        event.region1.add_match(
            match=event.region2,
            edge=event.edge
        )
        self.fill_system.set_region_growth(event.region1, new_growth=0)
        self.fill_system.set_region_growth(event.region2, new_growth=0)

        for r in [region1, region2]:
            self._shatter_descendants_into_matches_and_freeze(r)

    def extract_matching_and_reset_graph(self) -> Tuple[List['CompressedEdge'], int, int]:
        match_list = []
        total_weight = 0
        obs_mask = 0
        for d in self.detection_events:
            r = d.top_region()
            if r is not None:
                assert r.match is not None
                for m in r.to_subblossom_matches():
                    match_list.append(m.match.edge)
                    obs_mask ^= m.match.edge.obs_mask
                    total_weight += m.match.edge.distance
        return match_list, total_weight, obs_mask

    def shatter_and_match(self, max_depth: int = 1) -> None:
        """
        Shatter into matches, even if the algorithm hasn't terminated, for use in the demo. When this
        is used before the algorithm terminates, the final matching will not be minimum weight.
        """
        matches = []
        seen = set()
        depth = None if self.fill_system.has_valid_events_queued() else max_depth
        for r in list(self.iter_all_top_level_regions()):
            if r.match is not None and r.match.edge.loc_from.loc not in seen:
                seen.add(r.match.edge.loc_from.loc)
                area = list(r.iter_total_area())
                if r.match.region is not None:
                    seen.add(r.match.edge.loc_to.loc)
                    area += list(r.match.region.iter_total_area())
                for a in area:
                    a.invalidate_involved_schedule_items()
                for m in r.to_subblossom_matches(max_depth=depth):
                    if not m.blossom_parent and (m.match.region is None or not m.match.region.blossom_parent):
                        matches.append(m.match.edge)
                for a in area:
                    self.fill_system.reschedule_events_at_location(location_data=a)

        self.match_edges.extend(matches)

    def reset(self):
        self.detection_events = []
        self.fill_system.reset()

    def iter_all_top_level_regions(self) -> Iterator[GraphFillRegion]:
        seen = set()
        for d in self.detection_events:
            r = d.top_region()
            if r is not None and r.id not in seen:
                yield r

    def draw_areas(self, *, screen: 'pygame.Surface', scale: float):
        for k, r in enumerate(self.iter_all_top_level_regions()):
            tint = (int(127 * math.sin(k / 10) + 127),
                    int(127 * math.sin(6 + k / 5) + 127),
                    int(127 * math.sin(1 + k / 3) + 127),
                    )
            r.draw_area(screen=screen, scale=scale, time=self.fill_system.time, tint=tint)

    def draw_region_explored_edges(self, *, screen: 'pygame.Surface', scale: float):
        for r in self.iter_all_top_level_regions():
            r.draw_internal_graph_edges(screen=screen, scale=scale, time=self.fill_system.time)

    def draw_match_edges(self, *, screen: 'pygame.Surface', scale: float):
        matched = set()
        for d in self.detection_events:
            r = d.top_region()
            if r is not None and r.match is not None and r.id not in matched:
                matched.add(r.id)
                if r.match.region is not None:
                    matched.add(r.match.region.id)
                r.match.edge.draw_path(screen=screen, scale=scale, rgb=(0, 0, 0), width=5)

    def draw_internal_blossom_edges(self, *, screen: 'pygame.Surface', scale: float):
        seen = set()
        for r in self.iter_all_top_level_regions():
            seen.add(r.id)
            r.draw_blossom_cycle_edges(screen=screen, scale=scale)

    def draw_alternating_tree_edges(self, *, screen: 'pygame.Surface', scale: float):
        seen_outer_roots = set()
        for r in self.iter_all_top_level_regions():
            if r.alt_tree_node is not None:
                root = r.alt_tree_node.find_root()
                if root.outer_region.id not in seen_outer_roots:
                    seen_outer_roots.add(root.outer_region.id)
                    root.draw(screen=screen, scale=scale)

    def draw_detection_events(self, *, screen: 'pygame.Surface', scale: float):
        for e in self.detection_events:
            e.draw(screen=screen, scale=scale)

    def draw_final_matches(self, *, screen: 'pygame.Surface', scale: float):
        for e in self.match_edges:
            e.draw_path(screen=screen, scale=scale, rgb=(0, 0, 0), width=5)
