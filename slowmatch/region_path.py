from typing import List, Tuple, Optional, TYPE_CHECKING, Iterator
import dataclasses

from slowmatch.compressed_edge import CompressedEdge

if TYPE_CHECKING:
    from slowmatch.graph_fill_region import GraphFillRegion


@dataclasses.dataclass
class RegionEdge:
    region: 'GraphFillRegion'
    edge: Optional['CompressedEdge']
    
    def _eq_values(self):
        return (
            self.region.id,
            self.edge
        )

    def __eq__(self, other: 'RegionEdge') -> bool:
        return self._eq_values() == other._eq_values()

    def __str__(self) -> str:
        return f"({self.region.id}, ({self.edge.loc_from.loc}, {self.edge.loc_to.loc}))"


@dataclasses.dataclass
class RegionPath:
    edges: List[RegionEdge] = dataclasses.field(default_factory=lambda: [])

    def reversed(self) -> 'RegionPath':
        new_edges = []
        n = len(self.edges)
        for i in range(n - 1):
            current_edge = self.edges[n-i-1]
            next_edge = self.edges[n-i-2]
            new_edges.append(
                RegionEdge(
                    region=current_edge.region,
                    edge=next_edge.edge.reversed()
                )
            )
        new_edges.append(
            RegionEdge(
                region=self.edges[0].region,
                edge=None
            )
        )
        return RegionPath(edges=new_edges)

    def split_between_regions(
            self,
            start_region: 'GraphFillRegion',
            end_region: 'GraphFillRegion'
    ) -> Tuple['RegionPath', 'RegionPath']:
        """
            Returns the odd-length path from start_region to end_region inclusive, as well as the even-length
            path between start_region and end_region
        """
        regions = self.edges
        n = len(regions)
        i = next(i for i, r in enumerate(regions) if r.region is start_region)
        j = next(i for i, r in enumerate(regions) if r.region is end_region)
        regions = regions * 2
        result1 = regions[i: j + 1 + (n if j < i else 0)]
        result2 = regions[j: i + 1 + (n if i <= j else 0)]
        if len(result1) % 2 == 1:
            return RegionPath(result1), RegionPath(result2[1:-1])
        else:
            return RegionPath(result2).reversed(), RegionPath(result1[1:-1])

    def split_at_region(
            self,
            region: 'GraphFillRegion'
    ) -> 'RegionPath':
        n = len(self)
        if region is None:
            raise ValueError("Region cannot be None")
        if region not in [r.region for r in self]:
            raise ValueError(f"Region with id {region.id} not in path {self}")
        s = next(i for i, r in enumerate(self) if r.region is region)
        return (self * 2)[s + 1:s + n]

    def pairs_matched(self) -> Iterator['GraphFillRegion']:
        assert len(self) % 2 == 0
        for k in range(0, len(self), 2):
            a = self[k]
            b = self[k + 1]
            a.region.add_match(
                match=b.region,
                edge=a.edge
            )
            yield a.region

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RegionPath(self.edges[key])
        return self.edges[key]

    def __add__(self, other: 'RegionPath') -> 'RegionPath':
        return RegionPath(self.edges + other.edges)

    def __len__(self) -> int:
        return len(self.edges)

    def __iter__(self):
        return self.edges.__iter__()

    def __eq__(self, other) -> bool:
        return self.edges == other.edges

    def __mul__(self, other: int) -> 'RegionPath':
        return RegionPath(self.edges * other)

    def __bool__(self) -> bool:
        return bool(self.edges)
