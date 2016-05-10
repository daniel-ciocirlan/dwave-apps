__author__ = 'Daniel'

from collections import namedtuple
from numpy import prod

Point = namedtuple('Point', ['x', 'y'])

a = Point(2,3)
b = Point(3,4)

llist = [1,2,3,4,5,6]
mama = [0,0,0,1,1,0]

print list(tuple(llist))
print prod(llist)