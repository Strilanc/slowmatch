# coverage: ignore
import random
import sys
import time
import traceback
from typing import List, Tuple, Set, Optional, Iterator, Dict, Callable

import pygame

from slowmatch.graph_fill_region import GraphFillRegion
from slowmatch.graph_flooder import GraphFlooder
from slowmatch.mwpm import Mwpm
from slowmatch.exposed import graph_from_neighbors_and_boundary
from slowmatch.geometry import voronoi_from_points
from slowmatch.graph import LocationData


def complex_grid_neighbors(pos: complex) -> List[Tuple[int, int, complex]]:
    w1 = 141
    w2 = 100
    return [
        # (w1, 0, pos - 1j - 1),
        # (w1, 0, pos + 1j + 1),
        # (w1, 0, pos + 1j - 1),
        # (w1, 0, pos - 1j + 1),
        (w2, 0, pos - 1),
        (w2, 0, pos + 1),
        (w2, 0, pos - 1j),
        (w2, 0, pos + 1j),
        # (w2, 0, pos + 3 + 3j)
    ]


class Demo:
    def __init__(
            self,
            width: int = 1000,
            height: int = 800,
            detection_count: int = 20,
            error_rate: float = 0.07,
            scale: int = 60,
            time_scale: float = 40,
            default_case: str = "random-edges",
            area_visualisation: str = "polygon"
    ):
        self.width = int(width)
        self.height = int(height)
        self.detection_count = detection_count
        self.error_rate = error_rate
        self.scale = int(scale)
        self.time_scale = time_scale
        self.case_type = default_case
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.locs: List[complex] = []
        flooder: GraphFlooder = GraphFlooder(
            graph_from_neighbors_and_boundary(1 + 1j, complex_grid_neighbors, self.complex_grid_boundary)
        )
        self.mwpm: Mwpm = Mwpm(flooder=flooder)
        self.time_shift: int = 0
        self.paused_duration: float = 0
        self.paused_at_time: Optional[float] = None
        self.t0: Optional[float] = None
        self.num_nodes = sum(1 for n in self.mwpm.fill_system.graph.nodes.values()
                             if not self.complex_grid_boundary(n.loc))
        self.num_points_per_unit_distance = 40  # For voronoi diagram
        self.voronoi_regions = voronoi_from_points([l for _, _, l in self.points_on_graph()])
        self.point_to_blossom_depth: Dict[complex, (int, int)] = {}
        self.area_visualisation = area_visualisation

    def new_case(self):
        if self.case_type == "random-edges":
            self.locs = self.plausible_case()
        elif self.case_type == "random-nodes":
            det_count = min(self.detection_count, self.num_nodes)
            self.locs = self.generate_case()
        else:
            raise ValueError(f"Case type {self.case_type} not supported")

    def generate_case(self):
        locs = set()
        while len(locs) < self.detection_count:
            c = int(random.random() * self.width / self.scale) + int(random.random() * self.height / self.scale) * 1j
            if not self.complex_grid_boundary(c):
                locs.add(c)
        return list(locs)

    def plausible_case(self):
        kept = set()

        def flip(x):
            if self.complex_grid_boundary(x):
                return
            if x in kept:
                kept.remove(x)
            else:
                kept.add(x)

        for i in range(-1, self.width // self.scale + 1):
            for j in range(-1, self.height // self.scale + 1):
                c = i + j * 1j
                for _, _, other in complex_grid_neighbors(c):
                    if (c.real, c.imag) < (other.real, other.imag):
                        if random.random() < self.error_rate:
                            flip(c)
                            flip(other)
        return list(kept)

    def complex_grid_boundary(self, pos: complex) -> bool:
        return not 0 < pos.real < self.width / self.scale or not 0 < pos.imag < self.height / self.scale

    def points_on_graph(self) -> Iterator[Tuple['LocationData', int, complex]]:
        for node in self.mwpm.fill_system.graph.nodes.values():
            yield node, None, node.loc
        for node, i in self.mwpm.fill_system.graph.iter_all_edges():
            v = node.neighbors_with_boundary[i]
            vec = v.loc - node.loc
            num_points = int(self.num_points_per_unit_distance * abs(vec))
            step = vec / (num_points + 1)
            for i in range(self.num_points_per_unit_distance):
                pt = node.loc + step * (i + 1)
                yield node, i, round(pt.real, 8) + round(pt.imag, 8) * 1j

    def draw_voronoi(self, screen: pygame.Surface):
        self.point_to_blossom_depth = {}
        self.map_all_locations_to_blossom_depth()
        for loc, val in self.point_to_blossom_depth.items():
            if val is not None:
                depth, slope = val
                if loc in self.voronoi_regions:
                    reg = self.voronoi_regions[loc]
                    darken = 0.85 ** depth
                    if slope == 0:
                        color = (0, 255, 0)
                    elif slope > 0:
                        color = (255, 0, 0)
                    else:
                        color = (255, 255, 0)
                    color = tuple(int(darken * c) for c in color)
                    if len(reg) > 2:
                        poly = [(x.real * self.scale, x.imag * self.scale) for x in reg]
                        pygame.draw.polygon(surface=screen, color=color, points=poly, width=0)

    def map_region_locations_to_blossom_depth_and_growth(self, region: GraphFillRegion):
        recursive_depth = region.num_regions_above()

        for node in region.iter_total_area():
            for i in range(len(node.neighbors)):
                v = node.neighbors_with_boundary[i]
                vec = v.loc - node.loc
                if v.is_owned_by(region):
                    targ = v.loc
                else:
                    r = node.reached_from_source.cumulative_radius_to_region(region)(self.mwpm.fill_system.time)
                    covered_length = r - node.distance_from_source
                    effective_weight = (v.distance_from_source_almost_reached_from(node.reached_from_source)
                                        - node.distance_from_source)
                    f = covered_length / effective_weight if effective_weight > 0 else 1
                    targ = node.loc + f * vec
                num_points = int(self.num_points_per_unit_distance * abs(vec))
                step = vec / (num_points + 1)
                for i in range(self.num_points_per_unit_distance):
                    pt = node.loc + step * (i + 1)
                    pt = round(pt.real, 8) + round(pt.imag, 8) * 1j
                    if abs(pt - node.loc) < abs(targ - node.loc):
                        self.point_to_blossom_depth[pt] = (recursive_depth, region.radius.slope)

        for child in region.blossom_children:
            self.map_region_locations_to_blossom_depth_and_growth(region=child.region)

    def map_all_locations_to_blossom_depth(self):
        self.point_to_blossom_depth = {x: None for _, _, x in self.points_on_graph()}
        for r in self.mwpm.iter_all_top_level_regions():
            self.map_region_locations_to_blossom_depth_and_growth(region=r)

    def add_case_to_mwpm(self):
        for loc in self.locs:
            self.mwpm.add_detection_event(loc)

    def restart_mwpm(self):
        flooder = GraphFlooder(
            graph_from_neighbors_and_boundary(1 + 1j, complex_grid_neighbors, self.complex_grid_boundary)
        )
        self.mwpm = Mwpm(flooder=flooder)
        for loc in self.locs:
            self.mwpm.add_detection_event(loc)

    def reset_time(self):
        self.t0 = time.monotonic()
        self.paused_duration = 0
        self.paused_at_time = None
        self.time_shift = 0

    def draw_state(self):
        self.screen.fill((255, 255, 255))
        s1 = self.screen
        self.mwpm.fill_system.graph.draw(screen=s1, scale=self.scale)
        self.mwpm.draw_region_explored_edges(screen=s1, scale=self.scale)
        s2 = pygame.Surface((self.width, self.height))
        s2.fill((255, 255, 255))
        s2.set_alpha(150)
        if self.area_visualisation == "polygon":
            self.mwpm.draw_areas(screen=s2, scale=self.scale)
        elif self.area_visualisation == "voronoi":
            self.draw_voronoi(screen=s2)
        else:
            raise ValueError(f"Area visualisation type {self.area_visualisation} not supported.")
        s3 = pygame.Surface((self.width, self.height))
        s3.fill((1, 1, 1))
        s3.set_colorkey((1, 1, 1))
        self.mwpm.draw_match_edges(screen=s3, scale=self.scale)
        self.mwpm.draw_internal_blossom_edges(screen=s3, scale=self.scale)
        self.mwpm.draw_alternating_tree_edges(screen=s3, scale=self.scale)
        self.mwpm.draw_final_matches(screen=s3, scale=self.scale)
        self.mwpm.draw_detection_events(screen=s3, scale=self.scale)
        self.screen.blit(s2, (0, 0))
        self.screen.blit(s3, (0, 0))
        pygame.display.flip()

    def _handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.unicode == ' ':
                self.new_case()
                self.restart_mwpm()
                self.reset_time()
            if event.unicode == 'r':
                self.restart_mwpm()
                self.reset_time()
            if event.unicode == 'p':
                if self.paused_at_time is None:
                    self.paused_at_time = time.monotonic()
                else:
                    self.paused_duration += (time.monotonic() - self.paused_at_time)
                    self.paused_at_time = None
            if event.key == pygame.K_DOWN:
                self.restart_mwpm()
                self.time_shift -= 1
            if event.key == pygame.K_LEFT:
                self.restart_mwpm()
                self.time_shift -= 10
            if event.key == pygame.K_UP:
                self.time_shift += 1
            if event.key == pygame.K_RIGHT:
                self.time_shift += 10
            if event.unicode == "s":
                self.mwpm.shatter_and_match(max_depth=1)

        if event.type == pygame.QUIT:
            sys.exit()

    def loop(self):
        self.new_case()
        self.restart_mwpm()
        self.reset_time()
        try:
            while True:
                delay = self.paused_duration
                for event in pygame.event.get():
                    self._handle_event(event)
                if self.paused_at_time is not None:
                    delay += (time.monotonic() - self.paused_at_time)
                target_time = (time.monotonic() - self.t0 - delay) * self.time_scale + self.time_shift
                enter = time.monotonic()
                for _ in range(1000):
                    event = self.mwpm.fill_system.next_event(target_time)
                    if event is None:
                        self.mwpm.fill_system.time = target_time
                        break
                    else:
                        self.mwpm.process_event(event)
                        # flooder.time = event.time
                    if time.monotonic() > enter + 1:
                        break
                self.draw_state()
        except Exception:
            traceback.print_exc()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()


if __name__ == '__main__':
    demo = Demo(default_case="random-edges", error_rate=0.08, scale=50, area_visualisation="polygon")
    demo.loop()
