from typing import Optional

from slowmatch.flooder import Flooder


class RecordingFlooder(Flooder):
    def __init__(self, sub_flooder: Optional[Flooder] = None):
        self._next_id = 0
        self.sub_flooder = sub_flooder
        self.recorded_commands = []

    def create_region(self, location) -> int:
        if self.sub_flooder is None:
            result = self._next_id
            self._next_id += 1
        else:
            result = self.sub_flooder.create_region(location)
        self.recorded_commands.append(('create_region', location, result))
        return result

    def set_region_growth(self, region_id: int, *, new_growth: int):
        if self.sub_flooder is not None:
            self.sub_flooder.set_region_growth(region_id, new_growth=new_growth)
        self.recorded_commands.append(
            ('set_region_growth', region_id, new_growth))

    def create_blossom(self, contained_region_ids):
        if self.sub_flooder is None:
            result = self._next_id
            self._next_id += 1
        else:
            result = self.sub_flooder.create_blossom(contained_region_ids)
        self.recorded_commands.append(
            ('create_combined_region', tuple(contained_region_ids), result))
        return result

    def next_event(self, max_time=None):
        if self.sub_flooder is None:
            raise NotImplementedError()
        result = self.sub_flooder.next_event()
        self.recorded_commands.append(('next_event', result))
        return result


def test_record():
    a = RecordingFlooder()
    a.create_region('test')
    assert a.recorded_commands == [('create_region', 'test', 0)]
