import numbers
from typing import Union, Optional

import cirq


class Varying:
    """A value that is varying linearly over time."""

    T: 'Varying'

    def __init__(
        self,
        base: Union['Varying', float, int] = 0,
        *,
        slope: Union[int, float] = 0,
        base_time: Union[int, float] = 0,
    ):
        if isinstance(base, Varying):
            assert slope == 0 and base_time == 0
            self.slope = base.slope
            self._base = base._base
            self._base_time = base._base_time
        else:
            self.slope = float(slope)
            self._base = float(base)
            self._base_time = float(base_time)

    def __call__(self, time: float):
        return self._base + (time - self._base_time) * self.slope

    def __neg__(self):
        return self * -1

    def __mul__(self, other: Union[int, float]) -> 'Varying':
        if isinstance(other, (int, float)):
            return Varying(
                base_time=self._base_time, slope=self.slope * other, base=self._base * other
            )
        return NotImplemented

    def __truediv__(self, other: Union[int, float]) -> 'Varying':
        if isinstance(other, (int, float)):
            return Varying(
                base_time=self._base_time, slope=self.slope / other, base=self._base / other
            )
        return NotImplemented

    def __add__(self, other: Union[int, float, 'Varying']) -> 'Varying':
        if isinstance(other, (int, float)):
            return Varying(base_time=self._base_time, slope=self.slope, base=self._base + other)
        if isinstance(other, Varying):
            return Varying(
                base_time=self._base_time,
                slope=self.slope + other.slope,
                base=self._base + other(self._base_time),
            )
        return NotImplemented

    def __sub__(self, other: Union[int, float, 'Varying']) -> 'Varying':
        return self.__add__(-other)

    def __rsub__(self, other: Union[int, float, 'Varying']) -> 'Varying':
        return -self.__sub__(other)

    __rmul__ = __mul__
    __radd__ = __add__

    def __float__(self):
        if self.slope == 0:
            return float(self._base)
        return NotImplemented

    def __int__(self):
        if self.slope == 0:
            return int(self._base)
        return NotImplemented

    def __complex__(self):
        if self.slope == 0:
            return complex(self._base)
        return NotImplemented

    def __eq__(self, other):
        if self.slope == 0 and isinstance(other, numbers.Number):
            return self._base == other
        if isinstance(other, Varying):
            return self.slope == other.slope and self(0) == other(0)
        return NotImplemented

    def then_slope_at(self, *, time_of_change: float, new_slope: float) -> 'Varying':
        return Varying(base_time=time_of_change, base=self(time_of_change), slope=new_slope)

    def zero_intercept(self) -> Optional[float]:
        if self.slope == 0:
            return None
        return self._base_time - self._base / self.slope

    def _approx_eq_(self, other, atol: float):
        if self.slope == 0 and isinstance(other, numbers.Number):
            return cirq.approx_eq(self._base, other, atol=atol)
        if isinstance(other, Varying):
            return cirq.approx_eq(self.slope, other.slope, atol=atol) and cirq.approx_eq(
                self(0), other(0), atol=atol
            )
        return NotImplemented

    def __hash__(self):
        if self.slope == 0:
            return hash(self._base)
        return hash((Varying, self(0), self.slope))

    def __str__(self):
        return f'{self(0)} + T*{self.slope}'

    def __repr__(self):

        return f'({self(0)!r} + {self.slope!r}*Varying.T)'


Varying.T = Varying(slope=1)
