import dataclasses
from typing import TypeVar, List, Tuple, Optional, Dict, Union

from slowmatch.fill_system import FillSystem, Event, RegionHitRegionEvent, BlossomImplodeEvent

TLocation = TypeVar('TLocation')


@dataclasses.dataclass(eq=False)
class InnerNode(object):
    region_id: int
    parent: Optional['OuterNode']
    child: 'OuterNode'

    def all_matches_in_tree(self, *, out: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        if out is None:
            out = []
        out.append((self.region_id, self.child.region_id))
        for c in self.child.children:
            c.all_matches_in_tree(out=out)
        return out

    def __str__(self):
        return f'{self.region_id} ===> {self.child}'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.child == other.child


@dataclasses.dataclass
class PruneResult:
    orphans: List['InnerNode']  # Subtrees disconnected from the main tree by pruning the path.
    pruned_path_regions: List[int]  # Inner-outer node pairs


class OuterNode:
    def __init__(self, region_id: int):
        self.region_id = region_id
        self.parent: Optional['InnerNode'] = None
        self.children: List[InnerNode] = []

    def _tree_eq_helper(self, other, *, backwards: InnerNode):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.region_id != other.region_id:
            return False
        if (self.parent is None) != (other.parent is None):
            return False
        if self.parent is not backwards and self.parent is not None:
            if self.parent.region_id != other.parent.region_id:
                return False
            if (self.parent.parent is None) != (other.parent.parent is None):
                return False
            if self.parent.parent is not None and not self.parent.parent._tree_eq_helper(other.parent.parent, backwards=self.parent):
                return False
        if len(self.children) != len(other.children):
            return False
        for child, other_child in zip(self.children, other.children):
            if child is not backwards:
                if child.region_id != other_child.region_id:
                    return False
                if not child.child._tree_eq_helper(other_child.child, backwards=child):
                    return False
        return True

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._tree_eq_helper(other, backwards=None)

    def __ne__(self, other):
        return not self == other

    def all_region_ids_in_tree(self, *, out: Optional[List[int]] = None) -> List[int]:
        if out is None:
            out = []
        out.append(self.region_id)
        for c in self.children:
            out.append(c.region_id)
            c.child.all_region_ids_in_tree(out=out)
        return out

    def outer_ancestry(self, *, stop_before: Optional['OuterNode'] = None) -> List['OuterNode']:
        if self is stop_before:
            return []
        result = [self]
        while True:
            parent = result[-1].parent
            if parent is None or parent.parent is stop_before:
                break
            result.append(parent.parent)
        return result

    def become_root(self):
        """Performs a tree rotation that makes the current node the root."""
        if self.parent is None:
            return
        old_parent = self.parent
        old_grandparent = old_parent.parent

        self.children.append(old_parent)
        old_parent.child = old_grandparent
        old_parent.parent = self
        old_grandparent.children.remove(old_parent)
        old_grandparent.become_root()
        self.parent = None
        old_grandparent.parent = old_parent

    def most_recent_common_ancestor(self, other: 'OuterNode') -> 'OuterNode':
        seen = set(id(e) for e in self.outer_ancestry())
        for ancestor in other.outer_ancestry():
            if id(ancestor) in seen:
                return ancestor
        raise ValueError(f'No common ancestor between {self.region_id} and {other.region_id}.')

    def make_child_inner_outer(self, *, inner_region_id: int, outer_region_id: int) -> Tuple['InnerNode', 'OuterNode']:
        outer = OuterNode(region_id=outer_region_id)
        inner = InnerNode(parent=self, child=outer, region_id=inner_region_id)
        outer.parent = inner
        self.children.append(inner)
        return inner, outer

    def prune_upward_path_stopping_before(self, prune_parent: Optional['OuterNode'] = None) -> PruneResult:
        """Removes the path from `self` to just before `prune_parent` from the tree.

        Args:
            prune_parent: The node to stop just before. Must be an ancestor of `self`. Will not
                be removed from the tree.

        Returns:
            Children of nodes along the path, that are not themselves along the path, are reported
            in the `orphans` attribute of the result. Region ids of nodes removed by the pruning
            process are returned in the `pruned_path_region_ids` attribute of the result. The child
            of `prune_parent` that led to `self` is reported in the `top_pruned_node` attribute of
            the result.
        """
        orphans = []
        removed_regions = []
        for outer in self.outer_ancestry(stop_before=prune_parent):
            for child in outer.children:
                orphans.append(child)
                child.parent = None
            removed_regions.append(outer.region_id)
            inner = outer.parent
            if inner is not None:
                inner.parent.children = [
                    child for child in inner.parent.children if child is not inner
                ]
                inner.parent = None
                removed_regions.append(inner.region_id)
        return PruneResult(
            orphans=orphans,
            pruned_path_regions=removed_regions,
        )

    def __str__(self):
        indent1 = '+---'
        indent2 = '\n|   '
        childParagraphs = [indent1 + indent2.join(str(c).rstrip().split('\n'))
                           for c in self.children]
        result = str(self.region_id)
        if childParagraphs:
            result += '\n' + '\n'.join(childParagraphs).rstrip()
        return result


class MinWeightMatchingState:
    def __init__(self, fill_system: FillSystem):
        self.fill_system: FillSystem = fill_system
        self.tree_id_map: Dict[int, Union[InnerNode, OuterNode]] = {}
        self.match_map: Dict[int, int] = {}
        self.blossom_map: Dict[int, List[int]] = {}

    def add_region(self, region_id: int):
        self.tree_id_map[region_id] = OuterNode(region_id=region_id)

    def advance(self, max_time: Optional[float] = None) -> bool:
        event = self.fill_system.next_event(max_time=max_time)
        if event is None:
            return False
        self.process_event(event)
        return True

    def process_event(self, event: Event):
        if isinstance(event, BlossomImplodeEvent):
            self.handle_blossom_imploding(event)
        elif isinstance(event, RegionHitRegionEvent):
            if event.region1 in self.match_map or event.region2 in self.match_map:
                self.handle_tree_hitting_match(event)
            elif self.tree_id_map[event.region1].outer_ancestry()[-1] == self.tree_id_map[event.region2].outer_ancestry()[-1]:
                self.handle_tree_hitting_self(event)
            else:
                self.handle_tree_hitting_other_tree(event)
        else:
            raise NotImplementedError(f'Unrecognized event type "{type(event)}": {event!r}')

    def handle_blossom_imploding(self, event: BlossomImplodeEvent):
        blossom = self.tree_id_map[event.blossom_region]
        blossom_nodes = self.blossom_map[event.blossom_region]
        ancestor: OuterNode = self.tree_id_map[event.external_touching_region_1]
        descendent: OuterNode = self.tree_id_map[event.external_touching_region_2]
        i = blossom_nodes.index(event.internal_touching_region_1)
        j = blossom_nodes.index(event.internal_touching_region_2)
        evens, odds = cycle_split(blossom_nodes, i, j)
        odds = odds[::-1]
        if len(odds) % 2 == 0:
            odds, evens = evens, odds

        matches = evens[1:-1]
        for k in range(0, len(matches), 2):
            a = matches[k]
            b = matches[k + 1]
            self.match_map[a] = b
            self.match_map[b] = a

        ancestor.children = [e for e in ancestor.children if e is not blossom]
        outer = ancestor
        for k in range(len(odds), 2):
            inner, outer = ancestor.make_child_inner_outer(inner_region_id=odds[k],
                                                           outer_region_id=odds[k + 1])
            self.tree_id_map[inner.region_id] = inner
            self.tree_id_map[outer.region_id] = outer
        inner = InnerNode(region_id=odds[-1], parent=outer, child=descendent)
        self.tree_id_map[inner.region_id] = inner
        outer.children.append(inner)
        descendent.parent = inner

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
            outer_region_id=other_match,
            inner_region_id=match)
        self.tree_id_map[match] = inner
        self.tree_id_map[other_match] = outer

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


def cycle_split(items: List[T], i: int, j: int) -> Tuple[List[T], List[T]]:
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
    result1 = items[i:j + 1 + (n if j < i else 0)]
    result2 = items[j:i + 1 + (n if i <= j else 0)]
    return result1, result2
