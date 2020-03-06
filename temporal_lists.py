import random
import math


def _temporal_elem_key_fn(elem):
    return elem["start_time"]


class TemporalList(list):
    def __init__(self, elems=[], time_key="start_time"):
        self._time_key = time_key
        
        super(TemporalList, self).__init__(elems)
        self.__sort()
    
    def __sort(self):
        self.sort(key=_temporal_elem_key_fn)
    
    def append(self, elem):
        super(TemporalList, self).append(elem)
        self.__sort()
    
    def _prev_index(self, time):
        for i, elem in enumerate(reversed(self)):
            if elem[self._time_key] <= time:
                return len(self) - i - 1
        return None
    
    def _current_index(self, time):
        for i, elem in enumerate(self):
            if elem[self._time_key] > time:
                return None
            
            if elem[self._time_key] < time:
                continue
            
            return i
            
        return None
    
    def _next_index(self, time):
        for i, elem in enumerate(self):
            if elem[self._time_key] >= time:
                return i
        return None
    
    def prev(self, time, n=1):
        index = self._prev_index(time)
        if index is None:
            if n == 1:
                return None
            else:
                return []
        
        if n == 1:
            return self[index]
        else:
            return self[max(index - n + 1, 0):index + 1]
    
    def current(self, time):
        index = self._current_index(time)
        if index is None:
            return None
        
        return self[index]
    
    def next(self, time, n=1):
        index = self._next_index(time)
        if index is None:
            if n == 1:
                return None
            else:
                return []
        
        if n < 1:
            n = 1
        
        if n == 1:
            return self[index]
        else:
            return self[index:min(index + n, len(self))]

    def remove(self, time):
        index = self._current_index(time)
        del self[index]


class RangedTemporalList(TemporalList):
    def __init__(self, *args, end_time_key="end_time", **kwargs):
        self.__end_time_key = end_time_key
        super(RangedTemporalList, self).__init__(*args, **kwargs)
    
    def _prev_index(self, time):
        closest = (None, {self.__end_time_key: 0})
        for i, elem in enumerate(self):
            if closest[1][self.__end_time_key] < elem[self.__end_time_key] < time:
                closest = (i, elem)
        return closest[0]

    def _current_index(self, time):
        for i, elem in enumerate(self):
            if elem[self._time_key] <= time <= elem[self.__end_time_key]:
                return i
        return None
    
    def _next_index(self, time):
        closest = (None, {self._time_key: math.inf})
        for i, elem in enumerate(self):
            if time < elem[self._time_key] < closest[1][self._time_key]:
                closest = (i, elem)
        return closest[0]


class SequentialTemporalList(TemporalList):
    def __init__(self, *args, **kwargs):
        super(SequentialTemporalList, self).__init__(*args, **kwargs)
    
    def _prev_index(self, time):
        if len(self) < 2:
            return None
        
        for i, elem in enumerate(self):
            if time - elem[self._time_key] >= 0:
                continue
            
            if i < 2:
                return None
            
            return i - 2
        
        if time - self[-2][self._time_key] >= 0:
            return len(self) - 2
        
        return None
    
    def _current_index(self, time):
        for i, elem in enumerate(self):
            if time - elem[self._time_key] >= 0:
                continue
            
            if i < 1:
                return None
            
            return i - 1
        
        if len(self) > 0 and time - self[-1][self._time_key] >= 0:
            return len(self) - 1
        
        return None
    
    def _next_index(self, time):
        if len(self) < 1:
            return None
        
        for i, elem in enumerate(self):
            if time - elem[self._time_key] < 0:
                return i
        
        return None


if __name__ == "__main__":
    elems = [
        {
            "start_time": i * 2,
            "name": "element " + str(i * 2),
        } for i in range(5)
    ]
    random.shuffle(elems)
    l = SequentialTemporalList(elems)
    
    t = 5
    
    print(t, l.prev(t), l.current(t), l.next(t))
