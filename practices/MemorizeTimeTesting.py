import time
import functools
import collections


class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
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
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


def fib(n) :
    if n in (0, 1):
        return n
    return fib(n-1)+fib(n-2)

@ memoized
def fib2(n) :
    if n in (0, 1):
        return n
    return fib2(n-1)+fib2(n-2)

start = time.clock()
a = fib(40)
print(a)
end = time.clock()
cpu_time = end-start
print('cpu  time is %.3f' % cpu_time)
start = time.clock()
b = fib2(40)
print(b)
end = time.clock()
cpu_time = end-start
print('cpu  time after memoizing is %.3f' % cpu_time)