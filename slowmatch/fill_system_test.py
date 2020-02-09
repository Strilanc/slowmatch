from slowmatch.fill_system import VaryingRadius, FillSystem, TLocation


def test_varying_radius_radius_at():
    r1 = VaryingRadius(time0=2, growth=3, radius0=5)
    assert r1.radius_at(1) == 2
    assert r1.radius_at(2) == 5
    assert r1.radius_at(3) == 8


def test_varying_radius_sync():
    r1 = VaryingRadius(time0=2, growth=3, radius0=5)
    r1.sync(new_time=4, new_growth=-1)
    assert r1.radius0 == 11
    assert r1.time0 == 4
    assert r1.growth == -1

    r1.sync(new_time=5)
    assert r1.radius0 == 10
    assert r1.time0 == 5
    assert r1.growth == -1


def test_varying_radius_add():
    r1 = VaryingRadius(time0=2, growth=3, radius0=5)
    r2 = VaryingRadius(time0=7, growth=11, radius0=13)
    r3 = r1 + r2
    assert r1.radius_at(20) + r2.radius_at(20) == r3.radius_at(20)
    assert r1.radius_at(30) + r2.radius_at(30) == r3.radius_at(30)


class CommandRecordingFillSystem(FillSystem):
    def __init__(self):
        self._next_id = 0
        self.recorded_commands = []

    def create_region(self, location) -> int:
        result = self._next_id
        self._next_id += 1
        self.recorded_commands.append(('create_region', location, result))
        return result

    def set_region_growth(self, region_id: int, *, new_growth: int):
        self.recorded_commands.append(
            ('set_region_growth', region_id, new_growth))

    def create_blossom(self, contained_region_ids):
        result = self._next_id
        self._next_id += 1
        self.recorded_commands.append(
            ('create_combined_region', tuple(contained_region_ids), result))
        return result

    def next_event(self, max_time=None):
        raise NotImplementedError()
