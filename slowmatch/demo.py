# coverage: ignore
import random
import sys
import time
import traceback
from typing import List, Tuple, Union

import pygame

from slowmatch.alternating_tree import InnerNode
from slowmatch.circle_flooder import CircleFlooder
from slowmatch.graph_flooder import GraphFlooder
from slowmatch.mwpm import Mwpm


def complex_grid_neighbors(pos: complex) -> List[Tuple[int, complex]]:
    return [
        (141, pos - 1j - 1),
        (141, pos + 1j + 1),
        (100, pos - 1),
        (100, pos + 1),
        (100, pos - 1j),
        (100, pos + 1j),
    ]


def complex_grid_boundary(pos: complex) -> bool:
    return not 0 < pos.real < WIDTH / SCALE or not 0 < pos.imag < HEIGHT / SCALE


DETECTION_COUNT = 20
CIRCLES = False
SCALE = 20
TIME_SCALE = 100 / (100 if CIRCLES else 1)
ERROR_RATE = 0.03
WIDTH = 1500
HEIGHT = 1000


def generate_case(width: int, height: int):
    locs = set()
    while len(locs) < DETECTION_COUNT:
        c = int(random.random() * width / SCALE) + int(random.random() * height / SCALE) * 1j
        locs.add(c)
    return list(locs)


def plausible_case(width: int, height: int, p: float = ERROR_RATE):
    kept = set()

    def flip(x):
        if complex_grid_boundary(x):
            return
        if x in kept:
            kept.remove(x)
        else:
            kept.add(x)

    for i in range(-1, width // SCALE + 1):
        for j in range(-1, height // SCALE + 1):
            c = i + j * 1j
            for _, other in complex_grid_neighbors(c):
                if (c.real, c.imag) < (other.real, other.imag):
                    if random.random() < p:
                        flip(c)
                        flip(other)

    return list(kept)


def main():
    pygame.init()

    def restart(keep=False):
        nonlocal locs
        nonlocal flooder
        nonlocal mwpm
        nonlocal t0
        if not keep:
            if CIRCLES:
                locs = generate_case(WIDTH, HEIGHT)
            else:
                locs = plausible_case(WIDTH, HEIGHT)
        if CIRCLES:
            flooder = CircleFlooder(boundary_radius=WIDTH / SCALE)
        else:
            flooder = GraphFlooder(complex_grid_neighbors, complex_grid_boundary)
        mwpm = Mwpm(flooder=flooder)
        for loc in locs:
            mwpm.add_region(flooder.create_region(loc))
        t0 = time.monotonic()
        print(repr(locs))

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    locs = None
    flooder: CircleFlooder = None
    mwpm: Mwpm = None
    t0 = None
    restart()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.unicode == ' ':
                        restart()
                    if event.unicode == 'r':
                        restart(keep=True)
                    if event.unicode == 'f':
                        t0 -= 1
                if event.type == pygame.QUIT:
                    sys.exit()
            target_time = (time.monotonic() - t0) * TIME_SCALE
            enter = time.monotonic()
            for _ in range(1000):
                event = flooder.next_event(target_time)
                if event is None:
                    flooder.time = target_time
                    break
                else:
                    mwpm.process_event(event)
                    flooder.time = event.time
                if time.monotonic() > enter + 1:
                    break
            draw_state(screen, flooder, mwpm)
    except Exception:
        traceback.print_exc()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()


def draw_state(screen, flooder: Union[GraphFlooder[complex], CircleFlooder], mwpm: Mwpm):
    screen.fill((255, 255, 255))
    flooder.draw(screen=screen, scale=SCALE)
    for m1, m2 in mwpm.match_map.items():
        if m1 > m2:
            continue
        a, b = flooder.region_pair_to_line_segment_at_time(m1, m2, flooder.time,)
        pygame.draw.line(
            screen,
            (0, 0, 0),
            (int(a.real * SCALE + 0.5), int(a.imag * SCALE + 0.5)),
            (int(b.real * SCALE + 0.5), int(b.imag * SCALE + 0.5)),
            8,
        )

    for m1, m2 in mwpm.boundary_match_map.items():
        a, b = flooder.region_boundary_pair_to_line_segment_at_time(m1, m2, flooder.time,)
        pygame.draw.line(
            screen,
            (0, 0, 0),
            (int(a.real * SCALE + 0.5), int(a.imag * SCALE + 0.5)),
            (int(b.real * SCALE + 0.5), int(b.imag * SCALE + 0.5)),
            8,
        )

    for node in mwpm.tree_id_map.values():
        if node.parent is None:
            continue
        a, b = flooder.region_pair_to_line_segment_at_time(
            node.region_id, node.parent.region_id, flooder.time,
        )
        if isinstance(node, InnerNode):
            line_width = 8
            color = (0, 0, 0)
        else:
            line_width = 8
            color = (128, 128, 128)
        pygame.draw.line(
            screen,
            color,
            (int(a.real * SCALE + 0.5), int(a.imag * SCALE + 0.5)),
            (int(b.real * SCALE + 0.5), int(b.imag * SCALE + 0.5)),
            line_width,
        )
    pygame.display.flip()


if __name__ == '__main__':
    main()
