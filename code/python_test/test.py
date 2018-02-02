# coding=utf-8

import time
import functools

def time_log(func):
    # @functools.wraps(func)

    def wrapper(*args, **kwargs):
        start_time = time.time()
        print("{} start at: {}".format(func.__name__, time.strftime("%X", time.localtime())))
        back = func(*args, **kwargs)
        t = time.time() - start_time
        print("{} consume {}s".format(func.__name__, t))
        return back

    wrapper.__name__ == func.__name__
    return wrapper


@time_log
def fast(x, y):
    print("name is {}".format(fast.__name__))
    time.sleep(1.0012)
    return x + y;


@time_log
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z;

class Screen(object):
    def __init__(self, width, readonly):
        self._width = width
        self._readonly = readonly

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @width.deleter
    def width(self):
        del self._width
    @property
    def readonly(self):
        return self._readonly


if __name__ == "__main__":
    f = fast(11, 22)
    s = slow(11, 22, 33)
    int()
    if f != 33:
        print('测试失败!')
    elif s != 7986:
        print('测试失败!')

    s = Screen(100, 1)
    s.width = 100
    print(s.width)

    print(s.readonly)
    # s.readonly = 2
    delattr(s, 'width')
    # print(s.width)

    a = 99
    for a in range(5):
        if a == 4:
            print(a, '-> a in for-loop')
    print(a, '-> a in global')

    i = 1
    print([i for i in range(5)])
    print(i, '-> i in global')