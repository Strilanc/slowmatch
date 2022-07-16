import dataclasses
from typing import Optional, List, Generic, Union, Iterator, Tuple, TYPE_CHECKING, TypeVar

import cirq
from slowmatch.varying import Varying
from slowmatch.region_path import RegionPath
from slowmatch.compressed_edge import CompressedEdge
from slowmatch.geometry import get_unit_radius_polygon_around_node

if TYPE_CHECKING:
    from slowmatch.events import TentativeRegionShrinkEvent
    from slowmatch.graph import DetectorNode
    from slowmatch.alternating_tree import AltTreeNode
    import pygame

TLocation = TypeVar('TLocation')


@dataclasses.dataclass
class Match:
    region: Optional['GraphFillRegion']
    edge: 'CompressedEdge'


@cirq.value_equality(unhashable=True)
class GraphFillRegion(Generic[TLocation]):
    def __init__(
        self,
        *,
        id: int,
        source: Optional['DetectorNode'] = None,
        radius: Union[int, float, 'Varying'] = Varying.T,
        blossom_parent: Optional['GraphFillRegion'] = None,
        blossom_children: Optional[RegionPath] = None,
        alt_tree_node: Optional['AltTreeNode'] = None
    ):
        self.id = id
        self.source = source
        self.shell_area: List['DetectorNode'] = []
        self.radius = Varying(radius)
        self.blossom_children: Optional['RegionPath'] = (
            RegionPath() if blossom_children is None else blossom_children
        )
        self.alt_tree_node = alt_tree_node
        self.blossom_parent = blossom_parent
        self.shrink_event: Optional['TentativeRegionShrinkEvent'] = None
        self.match: Optional[Match] = None

    def iter_total_area(self) -> Iterator['DetectorNode']:
        for loc in reversed(self.shell_area):
            yield loc
        for child in self.blossom_children:
            yield from child.region.iter_total_area()

    def total_area_size(self) -> int:
        return len(self.shell_area) + sum(len(r.region.shell_area) for r in self.blossom_children)

    def is_matched_to_boundary(self) -> bool:
        return self.match is not None and self.match.region is None

    def num_regions_above(self) -> int:
        curr_region = self
        depth = 0
        while curr_region.blossom_parent is not None:
            curr_region = curr_region.blossom_parent
            depth += 1
        return depth

    def matched_to_region(self) -> bool:
        return self.match is not None and self.match.region is not None

    def invalidate_involved_schedule_items(self) -> None:
        if self.shrink_event is not None:
            self.shrink_event.invalidate()
            self.shrink_event = None

    def iter_all_sources(self) -> Iterator['DetectorNode']:
        if self.source is not None:
            yield self.source
        if self.blossom_children is not None:
            for child in self.blossom_children:
                yield from child.region.iter_all_sources()

    def top_region(self) -> 'GraphFillRegion':
        cur_region = self
        while cur_region.blossom_parent is not None:
            cur_region = cur_region.blossom_parent
        return cur_region

    def add_match(
            self,
            match: 'GraphFillRegion',
            edge: 'CompressedEdge'
    ):
        self.match = Match(region=match, edge=edge)
        match.match = Match(
            region=self,
            edge=edge.reversed()
        )
        self.alt_tree_node = None
        match.alt_tree_node = None

    def _value_equality_values_(self):
        src = self.source.loc if self.source is not None else None
        return self.id, src, self.radius, self.blossom_children

    def __repr__(self):
        return (
            f'GraphFillRegion('
            f'id={self.id!r}, '
            f'source={self.source}, '
            f'radius={self.radius!r}, '
            f'blossom_children={self.blossom_children!r})'
        )

    def blossom_depth(self):
        if not self.blossom_children:
            return 0
        else:
            return 1 + max(c.region.blossom_depth() for c in self.blossom_children)

    def cleanup_shell_area(self):
        for location_data in self.shell_area:
            location_data.cleanup()

    def shatter_into_matches_and_region(
            self,
            exclude_location: 'DetectorNode'
    ) -> Tuple[List['GraphFillRegion'], 'GraphFillRegion']:
        self.cleanup_shell_area()

        for child_edge in self.blossom_children:
            child_edge.region.blossom_parent = None

        exclude_subblossom = exclude_location.top_region()
        remaining_regions = self.blossom_children.split_at_region(region=exclude_subblossom)
        self.blossom_children = RegionPath()
        return list(remaining_regions.pairs_matched()), exclude_subblossom

    def to_subblossom_matches(
            self,
            max_depth: Optional[int] = None,
            depth: int = 0
    ) -> List['GraphFillRegion']:
        if max_depth is not None and depth >= max_depth:
            return [self]
        this_region = self
        match_region = self.match.region
        children_here = bool(this_region.blossom_children)
        children_there = match_region is not None and match_region.blossom_children
        if not children_here and not children_there:
            this_region.cleanup_shell_area()
            if this_region.match.region is not None:
                this_region.match.region.cleanup_shell_area()
            return [this_region]
        out = []
        if children_here:
            matches_1, region_1 = self.shatter_into_matches_and_region(exclude_location=self.match.edge.loc_from)
            region_1.match = self.match
            if match_region is not None:
                match_region.match.region = region_1
            this_region = region_1
            out.extend(m for match in matches_1 for m in match.to_subblossom_matches(
                max_depth=max_depth, depth=depth + 1))
        if children_there:
            matches_2, region_2 = match_region.shatter_into_matches_and_region(exclude_location=this_region.match.edge.loc_to)
            region_2.match = match_region.match
            this_region.match.region = region_2
            out.extend(m for match in matches_2 for m in match.to_subblossom_matches(
                max_depth=max_depth, depth=depth + 1))
        out.extend(m for m in this_region.to_subblossom_matches())
        return out

    def draw_area(self,
                  screen: 'pygame.Surface',
                  scale: float,
                  time: float,
                  tint: Tuple[float, float, float]) -> None:
        import pygame

        recursive_depth = self.num_regions_above()
        darken = 0.85 ** recursive_depth

        if self.radius.slope == 0:
            color = (0, 255, 0)
        elif self.radius.slope > 0:
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)
        color = tuple(a*0.7 + b*0.3 for a, b in zip(color, tint))
        color = tuple(int(darken*c) for c in color)

        for src in self.iter_all_sources():
            unit_poly = get_unit_radius_polygon_around_node(src)
            r = src.cumulative_radius_to_region(self)(time)
            poly = [(src.loc + (p * r)) * scale for p in unit_poly]
            poly = [(p.real, p.imag) for p in poly]
            if len(poly) > 2:
                pygame.draw.polygon(surface=screen, color=color, points=poly, width=0)

        for child in self.blossom_children:
            child.region.draw_area(screen=screen, scale=scale, time=time, tint=tint)

    def draw_internal_graph_edges(self, screen: 'pygame.Surface', scale: float, time: float) -> None:
        import pygame

        recursive_depth = self.num_regions_above()
        darken = 0.5 ** recursive_depth

        color = tuple(int(darken*c) for c in (100, 100, 100))

        for node in self.iter_total_area():
            for i in range(len(node.neighbors)):
                v = node.neighbors[i]
                if v is None:
                    continue
                if v.is_owned_by(self):
                    targ = v.loc
                else:
                    r = node.reached_from_source.cumulative_radius_to_region(self)(time)
                    covered_length = r - node.distance_from_source
                    effective_weight = (v.distance_from_source_almost_reached_from(node.reached_from_source)
                                        - node.distance_from_source)
                    f = covered_length/effective_weight if effective_weight > 0 else 1
                    targ = node.loc + f * (v.loc - node.loc)
                pygame.draw.line(
                    screen,
                    color,
                    (int(node.loc.real * scale + 0.5),
                     int(node.loc.imag * scale + 0.5)),
                    (int(targ.real * scale + 0.5),
                     int(targ.imag * scale + 0.5)),
                    width=2,
                )

        for child in self.blossom_children:
            child.region.draw_internal_graph_edges(screen=screen, scale=scale, time=time)

    def draw_blossom_cycle_edges(self, screen: 'pygame.Surface', scale: float) -> None:
        for e in self.blossom_children:
            e.edge.draw(screen=screen, scale=scale, rgb=(0, 0, 200), width=3)
            e.region.draw_blossom_cycle_edges(screen=screen, scale=scale)
