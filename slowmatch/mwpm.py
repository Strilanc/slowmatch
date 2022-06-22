from typing import TypeVar, List, Tuple, Dict, Union, Sequence, Any, Optional

from slowmatch.alternating_tree import InnerNode, OuterNode
from slowmatch.flooder import (
    Flooder,
    MwpmEvent,
    RegionHitRegionEvent,
    BlossomImplodeEvent,
    RegionHitBoundaryEvent,
)


class Mwpm:
    """The internal state of an embedded minimum weight perfect matching algorithm."""

    def __init__(self, flooder: Flooder):
        self.fill_system: Flooder = flooder
        self.tree_id_map: Dict[int, Union[InnerNode, OuterNode]] = {}
        self.match_map: Dict[int, int] = {}
        self.boundary_match_map: Dict[int, Any] = {}
        self.blossom_map: Dict[int, List[int]] = {}

    def add_region(self, region_id: int):
        self.tree_id_map[region_id] = OuterNode(region_id=region_id)

    def _region_tree_root(self, region_id: int) -> OuterNode:
        return self.tree_id_map[region_id].outer_ancestry()[-1]

    def _in_same_tree(self, region_id_1: int, region_id_2: int) -> bool:
        a = self._region_tree_root(region_id_1)
        b = self._region_tree_root(region_id_2)
        return a == b

    def process_event(self, event: MwpmEvent):
        if isinstance(event, BlossomImplodeEvent):
            self.handle_blossom_imploding(event)
        elif isinstance(event, RegionHitRegionEvent):
            if event.region1 in self.match_map or event.region2 in self.match_map:
                self.handle_tree_hitting_match(event)
            elif (
                event.region1 in self.boundary_match_map or event.region2 in self.boundary_match_map
            ):
                self.handle_tree_hitting_boundary_or_boundary_match(event)
            elif self._in_same_tree(event.region1, event.region2):
                self.handle_tree_hitting_self(event)
            else:
                self.handle_tree_hitting_other_tree(event)
        elif isinstance(event, RegionHitBoundaryEvent):
            self.handle_tree_hitting_boundary_or_boundary_match(event)
        else:
            raise NotImplementedError(f'Unrecognized event type "{type(event)}": {event!r}')

    def handle_blossom_imploding(self, event: BlossomImplodeEvent):
        blossom = self.tree_id_map[event.blossom_region_id]
        assert isinstance(blossom, InnerNode)
        del self.tree_id_map[blossom.region_id]

        # Find the ids of the inner regions touching the outer parent and child.
        out_child_id = blossom.child.region_id
        out_parent_id = blossom.parent.region_id
        in_child_id = -1
        in_parent_id = -1
        for in_id, out_id in event.in_out_touch_pairs:
            if out_id == out_child_id:
                in_child_id = in_id
            if out_id == out_parent_id:
                in_parent_id = in_id
        assert in_child_id != -1
        assert in_parent_id != -1

        # Find even and odd length paths from parent to child.
        inner_region_ids = self.blossom_map[event.blossom_region_id]
        evens, odds = cycle_split(
            inner_region_ids,
            inner_region_ids.index(in_parent_id),
            inner_region_ids.index(in_child_id),
        )
        odds = odds[::-1]
        if len(odds) % 2 == 0:
            odds, evens = evens, odds

        # The even length path becomes matches.
        matches = evens[1:-1]
        for k in range(0, len(matches), 2):
            a = matches[k]
            b = matches[k + 1]
            self.match_map[a] = b
            self.match_map[b] = a
            # Set region growth to zero (even though already zero) to ensure
            # these regions are rescheduled.
            self.fill_system.set_region_growth(a, new_growth=0)
            self.fill_system.set_region_growth(b, new_growth=0)

        # The odd length path is inserted into the alternating tree.
        ancestor: OuterNode = self.tree_id_map[out_parent_id]
        descendent: OuterNode = self.tree_id_map[out_child_id]
        ancestor.children = [e for e in ancestor.children if e is not blossom]
        cur_outer = ancestor
        for k in range(0, len(odds) - 1, 2):
            cur_inner, cur_outer = cur_outer.make_child_inner_outer(
                inner_region_id=odds[k], outer_region_id=odds[k + 1]
            )
            self.tree_id_map[cur_inner.region_id] = cur_inner
            self.tree_id_map[cur_outer.region_id] = cur_outer
            self.fill_system.set_region_growth(cur_inner.region_id, new_growth=-1)
            self.fill_system.set_region_growth(cur_outer.region_id, new_growth=+1)
        cur_inner = InnerNode(region_id=odds[-1], parent=cur_outer, child=descendent)
        self.fill_system.set_region_growth(cur_inner.region_id, new_growth=-1)
        self.tree_id_map[cur_inner.region_id] = cur_inner
        cur_outer.children.append(cur_inner)
        descendent.parent = cur_inner

    def handle_tree_hitting_match(self, event: RegionHitRegionEvent):
        """An outer node from an alternating tree hit a matched node from a match.

        This breaks the match apart, and adds its two nodes into the tree at the point of contact.

        Args:
            event: Event data including the two regions that are colliding.
        """
        if event.region1 in self.tree_id_map:
            node = self.tree_id_map[event.region1]
            match = event.region2
        else:
            node = self.tree_id_map[event.region2]
            match = event.region1
        other_match = self.match_map[match]
        del self.match_map[match]
        del self.match_map[other_match]
        self.fill_system.set_region_growth(match, new_growth=-1)
        self.fill_system.set_region_growth(other_match, new_growth=+1)
        inner, outer = node.make_child_inner_outer(
            outer_region_id=other_match, inner_region_id=match
        )
        self.tree_id_map[match] = inner
        self.tree_id_map[other_match] = outer

    def handle_tree_hitting_boundary_or_boundary_match(
        self, event: Union[RegionHitBoundaryEvent, RegionHitRegionEvent]
    ):
        """An outer node from an alternating tree hit a boundary or a node matched to a boundary.

        This shatters the node's tree, and matches the incoming node to the point of contact.

        Args:
            event: Event data including the region that is colliding.
        """
        if isinstance(event, RegionHitBoundaryEvent):
            # Match to boundary.
            node: OuterNode = self.tree_id_map[event.region]
            self.boundary_match_map[event.region] = event.boundary
        else:
            # Match to boundary associate.
            incoming = event.region1
            squished = event.region2
            if incoming in self.boundary_match_map:
                incoming, squished = squished, incoming
            assert squished in self.boundary_match_map
            node: OuterNode = self.tree_id_map[incoming]
            del self.boundary_match_map[squished]
            self.match_map[incoming] = squished
            self.match_map[squished] = incoming

        # Shatter the alternating tree into matches.
        self.fill_system.set_region_growth(node.region_id, new_growth=0)
        node.become_root()
        for k in node.all_region_ids_in_tree():
            del self.tree_id_map[k]
        for child in node.children:
            for a, b in child.all_matches_in_tree():
                self.match_map[a] = b
                self.match_map[b] = a
                self.fill_system.set_region_growth(a, new_growth=0)
                self.fill_system.set_region_growth(b, new_growth=0)

    def handle_tree_hitting_self(self, event: RegionHitRegionEvent):
        """Two outer nodes from an alternating tree have hit each other.

        This creates a cycle in the tree, requiring the creation of a blossom encompassing the two
        outer nodes and any nodes on the existing path between them.

        Args:
            event: Event data including the two regions that are colliding.
        """
        region1 = self.tree_id_map[event.region1]
        region2 = self.tree_id_map[event.region2]

        # Remove the path from the two nodes to their common ancestor from the tree.
        common_ancestor = region1.most_recent_common_ancestor(region2)
        p1 = region1.prune_upward_path_stopping_before(common_ancestor)
        p2 = region2.prune_upward_path_stopping_before(common_ancestor)

        # Determine what to add back into the tree.
        orphans: List[InnerNode] = [
            *common_ancestor.children,
            *p1.orphans,
            *p2.orphans,
        ]
        blossom_region_id_cycle = [
            *p1.pruned_path_regions,  # Travel up one path.
            common_ancestor.region_id,  # Switch directions where they meet.
            *p2.pruned_path_regions[::-1],  # Travel down the other path.
        ]
        blossom_id = self.fill_system.create_blossom(blossom_region_id_cycle)

        # Update the tree structure.
        blossom_node = OuterNode(region_id=blossom_id)
        blossom_node.children = orphans
        for orphan in orphans:
            orphan.parent = blossom_node
        blossom_node.parent = common_ancestor.parent
        if common_ancestor.parent is not None:
            common_ancestor.parent.child = blossom_node

        # Update the dictionaries.
        for e in blossom_region_id_cycle:
            del self.tree_id_map[e]
        self.blossom_map[blossom_id] = blossom_region_id_cycle
        self.tree_id_map[blossom_id] = blossom_node

    def handle_tree_hitting_other_tree(self, event: RegionHitRegionEvent):
        """Two outer nodes, from different alternating trees, have hit each other.

        This creates an augmenting path between the roots of the two trees, passing through the
        collision. After applying this augmenting path, both trees dissolve into matches.

        Args:
            event: Event data including the two regions that are colliding.
        """
        region1 = self.tree_id_map[event.region1]
        region2 = self.tree_id_map[event.region2]

        region1.become_root()
        region2.become_root()
        matches = [(region1.region_id, region2.region_id)]
        for r in [region1, region2]:
            for c in r.children:
                c.all_matches_in_tree(out=matches)
            for k in r.all_region_ids_in_tree():
                del self.tree_id_map[k]

        for a, b in matches:
            self.match_map[a] = b
            self.match_map[b] = a
            self.fill_system.set_region_growth(a, new_growth=0)
            self.fill_system.set_region_growth(b, new_growth=0)


T = TypeVar('T')


def cycle_split(items: Sequence[T], i: int, j: int) -> Tuple[List[T], List[T]]:
    """Splits a cyclical list into two parts terminating at the given boundaries.

    Each part includes both boundaries.

    Args:
        items: The cyclical list of items to split.
        i: Index of the first boundary.
        j: Index of the second boundary.

    Returns:
        A tuple containing the i-to-j part then the j-to-i part. When the two indices are equal,
        the i-to-j part is a singleton while the j-to-i part has length n+1.
    """
    n = len(items)
    i %= n
    j %= n
    items = list(items) * 2
    result1 = items[i : j + 1 + (n if j < i else 0)]
    result2 = items[j : i + 1 + (n if i <= j else 0)]
    return result1, result2
