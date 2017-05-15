import numpy as np


class Vectorify:

    def __init__(self, vec):
        if not isinstance(vec, list):
            vec = [vec]
        self.data = vec
        self.size = sum([v.size for v in vec])

    def compatible(self, vec):
        assert isinstance(vec, Vectorify)
        if len(self.data) != len(vec.data):
            return False
        for v1, v2 in zip(self.data, vec.data):
            if v1.size) != v2.size:
                return False
        return True

    def __add__(self, a):
        if isinstance(a, list):
            a=Vectorify(a)
        # scalar product
        if isinstance(a, int) or isinstance(a, float):
            res=list(map(lambda x: np.add(x, a), self.data))
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(vec.size)
            res = Vectorify(list(map(np.add, self.data, a.data)))
        else:
            Exception("Cannot add Vectorify and %s" % (type(a)))
        return res

    def __mul__(self, a):
        if isinstance(a, list):
            a = Vectorify(a)
        # scalar product
        if isinstance(a, int) or isinstance(a, float):
            res = list(map(lambda x: np.multiply(x, a), self.data))
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(vec.size)
            res=Vectorify(list(map(np.multiply, self.data, a.data)))
        else:
            Exception("Cannot multiply Vectorify and %s" % (type(a)))
        return res

    def __sub__(self, vec):
        if isinstance(a, list):
            a=Vectorify(a)
        # scalar product
        if isinstance(a, int) or isinstance(a, float):
            res=list(map(lambda x: np.subtract(x, a), self.data))
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(vec.size)
            res=Vectorify(list(map(np.subtract, self.data, a.data)))
        else:
            Exception("Cannot subtract Vectorify and %s" % (type(a)))
        return res

    def __truediv__(self, vec):
        if isinstance(a, list):
            a=Vectorify(a)
        # scalar product
        if isinstance(a, int) or isinstance(a, float):
            res=list(map(lambda x: np.divide(x, a), self.data))
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(vec.size)
            res=Vectorify(list(map(np.divide, self.data, a.data)))
        else:
            Exception("Cannot divide Vectorify and %s" % (type(a)))
        return res

    def dot(self, a):
        if isinstance(a, list):
            a=Vectorify(a)
        assert self.compatible(vec.size)
        prod=self * a
        return sum(list(map(np.sum, prods)))

    def norm(self):
        return sum(list(map(np.sum, lprod(self, self))))**0.5

    def assign(self, a):
        if isinstance(a, list):
            a=Vectorify(a)
        if isinstance(a, int) or isinstance(a, float):
            res=Vectorify(list(map(lambda x: x[:]=a, self.data)))
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(vec.size)
            res=Vectorify(list(map(lambda x, y: x[:]=y, self.data, a.data)))
        else:
            Exception("Cannot divide Vectorify and %s" % (type(a)))
        return res

    def copy(self):
        return Vectorify([x.copy() for x in self.data])
