import sys
import time

import pygame

from slowmatch.euclidean_fill import InefficientEuclideanFillSystem


def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    state = InefficientEuclideanFillSystem()
    state.time = -30
    state.create_region(10 + 50j)
    state.create_region(300 + 250j)
    state.create_region(200 + 150j)
    state.time = 0
    state.create_blossom([0, 1])

    t0 = time.monotonic()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        target_time = (time.monotonic() - t0) * 10
        screen.fill((255, 255, 255))
        event = state.next_event()
        if event.time > target_time:
            state.time = target_time
        state.draw(screen=screen)
        pygame.display.flip()


if __name__ == '__main__':
    main()
