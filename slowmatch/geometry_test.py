import pytest

from slowmatch.geometry import is_left_turn, graham_scan


def test_is_left_turn():
    assert is_left_turn(1, 2, 3) in [False, True]
    assert not is_left_turn(0, 0 + 1j, -1 + 1j)
    assert is_left_turn(1, 2, 3 - 1j)


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
