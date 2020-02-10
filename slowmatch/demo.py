import math
import random
import sys
import time

import pygame

from slowmatch.alternating_tree import InnerNode
from slowmatch.circle_flooder import CircleFlooder
from slowmatch.flooder import Flooder
from slowmatch.mwpm import Mwpm


def main():
    pygame.init()

    def restart(keep=False):
        nonlocal locs
        nonlocal flooder
        nonlocal mwpm
        nonlocal t0
        if not keep:
            locs = [
                random.random() * width + random.random() * 1j * height
                for _ in range(100)
            ]
        flooder = CircleFlooder()
        mwpm = Mwpm(flooder=flooder)
        for loc in locs:
            mwpm.add_region(flooder.create_region(loc))
        t0 = time.monotonic()
        print(repr(locs))

    width = 1500
    height = 1000
    screen = pygame.display.set_mode((width, height))
    time_scale = 100000000
    locs = None
    flooder: CircleFlooder = None
    mwpm: Mwpm = None
    t0 = None
    restart()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.unicode == ' ':
                    restart()
                if event.unicode == 'r':
                    restart(keep=True)
            if event.type == pygame.QUIT:
                sys.exit()
        target_time = (time.monotonic() - t0) * time_scale
        for _ in range(1):
            event = flooder.next_event(target_time)
            if event is None:
                flooder.time = target_time
                break
            else:
                mwpm.process_event(event)
                flooder.time = event.time
        draw_state(screen, flooder, mwpm)


def draw_state(screen, flooder: CircleFlooder, mwpm: Mwpm):
    screen.fill((255, 255, 255))
    flooder.draw(screen=screen)
    for m1, m2 in mwpm.match_map.items():
        if m1 > m2:
            continue
        a, b = flooder.region_pair_to_line_segment_at_time(
            m1,
            m2,
            flooder.time,
        )
        pygame.draw.line(screen,
             (0, 0, 0),
             (int(a.real), int(a.imag)),
             (int(b.real), int(b.imag)),
             3)
    for node in mwpm.tree_id_map.values():
        if node.parent is None:
            continue
        a, b = flooder.region_pair_to_line_segment_at_time(
            node.region_id,
            node.parent.region_id,
            flooder.time,
        )
        if isinstance(node, InnerNode):
            line_width = 3
            color = (0, 0, 0)
        else:
            line_width = 1
            color = (128, 128, 128)
        pygame.draw.line(screen,
                         color,
                         (int(a.real), int(a.imag)),
                         (int(b.real), int(b.imag)),
                         line_width)
    pygame.display.flip()


if __name__ == '__main__':
    main()
