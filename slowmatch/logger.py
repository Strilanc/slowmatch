from collections import Counter
from typing import List


class Logger:
    def __init__(self, enabled: bool = False):
        self.area_counter: Counter = Counter()
        self.boundary_counter: Counter = Counter()
        self.enabled: bool = enabled
        self.event_list: List[int] = []
        self.blossom_created_stats_list = []
        self.blossom_implosion_stats_list = []
        self.num_regions = 0

    def log_area_set(self, area_size: int):
        self.area_counter.update([area_size])
        self.event_list.append(area_size)

    def log_blossom_created(self, blossom_size: int, blossom_depth: int):
        self.blossom_created_stats_list.append((blossom_size, blossom_depth))

    def log_blossom_implosion(self, blossom_size: int, blossom_depth: int):
        self.blossom_implosion_stats_list.append((blossom_size, blossom_depth))

    def log_region_created(self):
        self.num_regions += 1
