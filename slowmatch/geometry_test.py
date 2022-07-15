import pytest

from slowmatch.geometry import ccw, graham_scan, sort_counter_clockwise_around_source


def test_ccw():
    assert ccw(1, 2, 3) == 0
    assert ccw(0, 0 + 1j, -1 + 1j) > 0
    assert ccw(1, 2, 3 - 1j) < 0


@pytest.mark.parametrize(
    "points,expected",
    [
        (
            [-1, 1j, 1, 0, -1j],
            [-1j, 1, 1j, -1]
        )
    ]
)
def test_graham_scan(points, expected):
    assert graham_scan(points) == expected


def test_sort_counter_clockwise_around_source():
    res = sort_counter_clockwise_around_source([-1 - 1j, 1 + 1j, -1 + 1j, 1 - 1j], source=0)
    assert res == [-1 -1j, 1 - 1j, 1 + 1j, -1 + 1j]
