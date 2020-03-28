# coverage: ignore
import random
from typing import List, Tuple

from slowmatch.graph_flooder import GraphFlooder
from slowmatch.mwpm import Mwpm


def complex_grid_neighbors_old(pos: complex) -> List[Tuple[int, complex]]:
    return [
        (20, pos - 1j),
        (20, pos + 1j),
        (22, pos - 1j - 1),
        (22, pos + 1j + 1),
        (10, pos - 1),
        (10, pos + 1),
    ]


def complex_grid_neighbors(pos: complex) -> List[Tuple[int, complex]]:
    return [
        (141, pos - 1j - 1),
        (141, pos + 1j + 1),
        (100, pos - 1),
        (100, pos + 1),
    ]


def generate_case(n: int, width: int, height: int, scale: int):
    assert n % 2 == 0
    locs = set()
    while len(locs) < n:
        c = int(random.random() * width) * scale + int(random.random() * height) * 1j * scale
        locs.add(c)
    return list(locs)


def run_locs(locs: List[complex], neigh):
    print(repr(locs))

    flooder = GraphFlooder(neigh)
    mwpm = Mwpm(flooder=flooder)
    for loc in locs:
        mwpm.add_region(flooder.create_region(loc))

    try:
        while True:
            event = flooder.next_event()
            if event is None:
                break
            mwpm.process_event(event)
            flooder.time = event.time
    except Exception:
        raise


def main():
    while True:
        neigh = complex_grid_neighbors
        case = generate_case(10, width=20, height=20, scale=3)
        run_locs(case, neigh=neigh)


if __name__ == '__main__':
    main()
