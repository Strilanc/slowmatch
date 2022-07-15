import dataclasses
from typing import List, Tuple, Optional, TYPE_CHECKING
from slowmatch.graph_fill_region import GraphFillRegion
from slowmatch.region_path import RegionPath, RegionEdge
from slowmatch.compressed_edge import CompressedEdge

if TYPE_CHECKING:
    import pygame


@dataclasses.dataclass
class AltTreeEdge:
    node: 'AltTreeNode'
    edge: 'CompressedEdge'


@dataclasses.dataclass
class AltTreePruneResult:
    orphans: List['AltTreeEdge']  # Subtrees disconnected from the main tree by pruning the path.
    pruned_path_regions: 'RegionPath'  # Inner-outer node pairs


@dataclasses.dataclass
class AltTreeNode:
    inner_region: Optional['GraphFillRegion']
    outer_region: 'GraphFillRegion'
    parent: Optional['AltTreeEdge'] = None
    children: List['AltTreeEdge'] = dataclasses.field(default_factory=lambda: [])
    inner_outer_edge: Optional[CompressedEdge] = None

    def all_matches_in_tree(
            self, *, out: Optional[List[Tuple['GraphFillRegion', 'GraphFillRegion']]] = None
            ) -> List[Tuple['GraphFillRegion', 'GraphFillRegion']]:
        if out is None:
            out = []
        out.append((self.inner_region, self.outer_region))
        for c in self.children:
            c.node.all_matches_in_tree(out=out)
        return out

    def shatter_into_matches(
            self, *, out: Optional[List[Tuple['GraphFillRegion', 'GraphFillRegion']]] = None,
            recursion_depth: int = 0
            ) -> List['GraphFillRegion']:
        if out is None:
            out = []
        if self.inner_region is not None:
            self.parent = None
            self.inner_region.add_match(
                match=self.outer_region,
                edge=self.inner_outer_edge
            )
            out.append(self.inner_region)
        for c in self.children:
            c.node.shatter_into_matches(out=out, recursion_depth=recursion_depth + 1)
        self.children = []
        return out

    def as_tuple(self):
        if self.inner_region is None:
            this_node = ((None, None), (self.outer_region.id, None))
        else:
            this_node = ((self.inner_region.id, self.inner_outer_edge.source1.loc),
                         (self.outer_region.id, self.inner_outer_edge.source2.loc))
        return this_node, tuple(c.node.as_tuple() for c in self.children)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.inner_region != other.inner_region or self.outer_region != other.outer_region:
            return False
        return self.find_root().as_tuple() == other.find_root().as_tuple()

    def __ne__(self, other):
        return not self == other

    def outer_ancestry(self, *, stop_before: Optional['AltTreeNode'] = None) -> List['AltTreeNode']:
        if self is stop_before:
            return []
        result = [self]
        while True:
            current_node = result[-1]
            if current_node.inner_region is None or current_node.parent.node is stop_before:
                break
            result.append(current_node.parent.node)
        return result

    def become_root(self):
        if self.inner_region is None:
            assert self.parent is None
            return
        assert self.parent is not None
        old_parent = self.parent.node
        old_parent.become_root()
        assert self.parent.node.inner_region is None
        self.parent.node.inner_region = self.inner_region
        self.parent.node.inner_outer_edge = self.parent.edge
        self.inner_region = None
        self.parent.node.children = [e for e in self.parent.node.children if e.node is not self]
        self.parent = None
        self.add_child(
            AltTreeEdge(
                node=old_parent,
                edge=self.inner_outer_edge
            )
        )

    def add_child(self, child: 'AltTreeEdge'):
        self.children.append(child)
        child.node.parent = AltTreeEdge(
            node=self,
            edge=child.edge.reversed()
        )

    def validate_parent_child(self):
        for c in self.children:
            assert c.node.parent.node == self
            c.node.validate_parent_child()

    def find_root(self) -> 'AltTreeNode':
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent.node
        return current_node

    def most_recent_common_ancestor(self, other: 'AltTreeNode') -> 'AltTreeNode':
        seen = set(id(e) for e in self.outer_ancestry())
        for ancestor in other.outer_ancestry():
            if id(ancestor) in seen:
                return ancestor
        raise ValueError(f'No common ancestor between ({self.inner_region},{self.outer_region}) '
                         f'and ({self.inner_region},{self.outer_region}).')

    def in_same_tree_as(self, other: 'AltTreeNode') -> bool:
        return self.find_root() is other.find_root()

    def __str__(self):
        indent1 = '+---'
        indent2 = '\n|   '
        child_paragraphs = [
            indent1 + indent2.join(str(c.node).rstrip().split('\n')) for c in self.children
        ]
        if self.inner_region is None:
            result = f"{self.outer_region.id}"
        else:
            result = f"{self.inner_region.id} ===> {self.outer_region.id}"
        if child_paragraphs:
            result += '\n' + '\n'.join(child_paragraphs).rstrip()
        return result

    def make_child_inner_outer(
            self,
            *,
            inner_graph_fill_region: 'GraphFillRegion',
            outer_graph_fill_region: 'GraphFillRegion',
            inner_outer_edge: CompressedEdge,
            child_edge: CompressedEdge
    ) -> 'AltTreeNode':
        node = AltTreeNode(inner_region=inner_graph_fill_region,
                           outer_region=outer_graph_fill_region,
                           inner_outer_edge=inner_outer_edge
                           )
        inner_graph_fill_region.alt_tree_node = node
        outer_graph_fill_region.alt_tree_node = node
        self.add_child(
            AltTreeEdge(
                node=node,
                edge=child_edge
            )
        )
        return node

    def prune_upward_path_stopping_before(
        self, prune_parent: Optional['AltTreeNode'] = None
    ) -> AltTreePruneResult:
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
        for node in self.outer_ancestry(stop_before=prune_parent):
            for child in node.children:
                orphans.append(child)
                child.node.parent = None
            removed_regions.append(
                RegionEdge(
                    region=node.outer_region,
                    edge=node.inner_outer_edge.reversed()
                )
            )
            node.outer_region.alt_tree_node = None
            if node.parent is not None:
                assert node.inner_region is not None
                removed_regions.append(
                    RegionEdge(
                        region=node.inner_region,
                        edge=node.parent.edge
                    )
                )
                node.parent.node.children = [c for c in node.parent.node.children if c.node is not node]
                node.parent = None
                node.inner_region.alt_tree_node = None
        return AltTreePruneResult(orphans=orphans, pruned_path_regions=RegionPath(removed_regions))

    def draw(self, screen: 'pygame.Surface', scale: float):
        if self.inner_outer_edge is not None:
            self.inner_outer_edge.draw(screen=screen, scale=scale, rgb=(80, 80, 80))
        for child in self.children:
            child.edge.draw(screen=screen, scale=scale, rgb=(150, 150, 150))
            child.node.draw(screen=screen, scale=scale)
