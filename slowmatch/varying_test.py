import cirq
import pytest

from slowmatch import Varying


def test_init():
    z = Varying()
    assert z._base_time == 0
    assert z._base == 0
    assert z.slope == 0
    assert z(0) == 0
    assert z(1) == 0

    z = Varying(base=2, base_time=3, slope=5)
    assert z._base_time == 3
    assert z._base == 2
    assert z.slope == 5
    assert z(0) == -13
    assert z(1) == -8

    z = Varying(z)
    assert z._base_time == 3
    assert z._base == 2
    assert z.slope == 5
    assert z(0) == -13
    assert z(1) == -8


def test_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        Varying(base_time=0, slope=1, base=0),
        Varying(base_time=1, slope=1, base=1),
        Varying(Varying(base_time=0, slope=1, base=0)),
    )
    eq.add_equality_group(
        Varying(base_time=0, slope=-1, base=0),
        Varying(base_time=1, slope=-1, base=-1),
        -Varying.T,
    )
    eq.add_equality_group(
        Varying(base_time=0, slope=1, base=1),
        Varying(base_time=1, slope=1, base=2),
    )
    eq.add_equality_group(
        Varying(base_time=0, slope=0, base=50),
        Varying(base_time=10000, slope=0, base=50),
        50,
        50.0,
        50.0 + 0j,
    )
    eq.add_equality_group(
        Varying(base_time=0, slope=0, base=60),
        Varying(base_time=10000, slope=0, base=60),
        60,
    )


def test_approximate_equality():
    assert cirq.approx_eq(Varying(5), Varying(5.01), atol=0.1)
    assert not cirq.approx_eq(Varying(5), Varying(5.01), atol=0.001)
    assert cirq.approx_eq(Varying(5, slope=4), Varying(5, slope=4.01), atol=0.1)
    assert not cirq.approx_eq(Varying(5, slope=4), Varying(5, slope=4.01), atol=0.001)


def test_arithmetic():
    a = Varying(base_time=2, slope=3, base=5)
    b = Varying(base_time=7, slope=11, base=13)

    add = a + b
    assert add._base_time == a._base_time
    assert a(20) + b(20) == add(20)
    assert a(30) + b(30) == add(30)

    sub = a - b
    assert add._base_time == a._base_time
    assert a(20) - b(20) == sub(20)
    assert a(30) - b(30) == sub(30)

    neg = -a
    assert neg._base_time == a._base_time
    assert a(20) == -neg(20)
    assert a(30) == -neg(30)

    offset1 = a + 2
    offset2 = 2 + a
    assert add._base_time == a._base_time
    assert a(20) + 2 == offset1(20) == offset2(20)
    assert a(30) + 2 == offset1(30) == offset2(30)

    dif1 = a - 2
    dif2 = 2 - a
    assert add._base_time == a._base_time
    assert a(20) - 2 == dif1(20) == -dif2(20)
    assert a(30) - 2 == dif1(30) == -dif2(30)

    double1 = a * 2
    double2 = 2 * a
    assert double1._base_time == a._base_time
    assert double2._base_time == a._base_time
    assert double1(20) == double2(20) == a(20) * 2
    assert double1(30) == double2(30) == a(30) * 2


def test_scalar():
    with pytest.raises(TypeError):
        _ = int(Varying(slope=1))
    with pytest.raises(TypeError):
        _ = float(Varying(slope=1))
    with pytest.raises(TypeError):
        _ = complex(Varying(slope=1))
    assert float(Varying(2)) == 2
    assert int(Varying(2)) == 2
    assert complex(Varying(2)) == 2
    assert isinstance(float(Varying(2)), float)
    assert isinstance(int(Varying(2)), int)
    assert isinstance(complex(Varying(2)), complex)


def test_then_slope_at():
    a = Varying(base_time=2, slope=3, base=5)
    b = a.then_slope_at(time_of_change=7, new_slope=11)
    assert b._base_time == 7
    assert b(7) == a(7)
    assert b.slope == 11


def test_repr():
    cirq.testing.assert_equivalent_repr(
        Varying(base_time=2, slope=3, base=5),
        setup_code='from slowmatch import Varying')


def test_str():
    assert str(5.0 + Varying.T * 3) == '5.0 + T*3.0'


def test_zero_intercept():
    assert Varying.T.zero_intercept() == 0
    assert (Varying.T + 5).zero_intercept() == -5
    assert (-Varying.T*2 + 5).zero_intercept() == 2.5
    assert (-Varying.T*0 + 5).zero_intercept() is None
    assert Varying(base_time=5, base=3, slope=2).zero_intercept() == 3.5
    assert (Varying.T*0).zero_intercept() is None
