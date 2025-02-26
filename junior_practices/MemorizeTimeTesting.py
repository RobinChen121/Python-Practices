import time
import functools
from collections.abc import Hashable
from functools import lru_cache

class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def fib(n):
    if n in (0, 1):
        return n
    return fib(n - 1) + fib(n - 2)


@memoized
def fib2(n):
    if n in (0, 1):
        return n
    return fib2(n - 1) + fib2(n - 2)


@lru_cache()
def fib3(n):
    if n in (0, 1):
        return n
    return fib3(n - 1) + fib3(n - 2)


# start = time.process_time()
# a = fib(50)
# print(a)
# end = time.process_time()
# cpu_time = end-start
# print('cpu  time is %.3f' % cpu_time)

n = 2500
# start = time.process_time()
# b = fib2(n)
# end = time.process_time()
# print(b)
# cpu_time = end - start
# print("cpu  time after memorizing is %.4f" % cpu_time)



start = time.process_time()
b = fib3(n)
end = time.process_time()
print(b)
cpu_time = end - start
print("n is %d, cpu  time after memorizing is %.4f" % (n,cpu_time))
