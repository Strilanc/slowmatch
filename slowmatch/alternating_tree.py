import dataclasses
from typing import List, Tuple, Optional


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
    orphans: List[InnerNode]  # Subtrees disconnected from the main tree by pruning the path.
    pruned_path_regions: List[int]  # Inner-outer node pairs


class OuterNode:
    def __init__(self, region_id: int):
        self.region_id = region_id
        self.parent: Optional['InnerNode'] = None
        self.children: List[InnerNode] = []

    def _tree_eq_helper(self, other, *, backwards: Optional[InnerNode]):
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
        child_paragraphs = [indent1 + indent2.join(str(c).rstrip().split('\n'))
                           for c in self.children]
        result = str(self.region_id)
        if child_paragraphs:
            result += '\n' + '\n'.join(child_paragraphs).rstrip()
        return result
