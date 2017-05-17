import numpy as np
import pickle


class Vectorify:

    def __init__(self, vec):
        if isinstance(vec, Vectorify):
            self.data = [x.copy() for x in vec.data]
        elif isinstance(vec, list):
            self.data = vec
        elif isinstance(vec, str):
            self.data = pickle.load(open( vec, "rb" ))
        elif not isinstance(vec, list):
            self.data = [vec]
        self.size = sum([v.size for v in self.data])

    def compatible(self, vec):
        assert isinstance(vec, Vectorify)
        if len(self.data) != len(vec.data):
            return False
        for v1, v2 in zip(self.data, vec.data):
            if v1.size != v2.size:
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
            assert self.compatible(a)
            res = list(map(np.add, self.data, a.data))
        else:
            Exception("Cannot add Vectorify and %s" % (type(a)))
        return Vectorify(res)

    def __radd__(self, a):
        return self + a

    def __mul__(self, a):
        if isinstance(a, list):
            a = Vectorify(a)
        # scalar product
        if isinstance(a, int) or isinstance(a, float):
            res = list(map(lambda x: np.multiply(x, a), self.data))
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(a)
            res=list(map(np.multiply, self.data, a.data))
        else:
            Exception("Cannot multiply Vectorify and %s" % (type(a)))
        return Vectorify(res)

    def __sub__(self, a):
        if isinstance(a, list):
            a=Vectorify(a)
        # scalar product
        if isinstance(a, int) or isinstance(a, float):
            res=list(map(lambda x: np.subtract(x, a), self.data))
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(a)
            res=list(map(np.subtract, self.data, a.data))
        else:
            Exception("Cannot subtract Vectorify and %s" % (type(a)))
        return Vectorify(res)

    def __truediv__(self, a):
        if isinstance(a, list):
            a=Vectorify(a)
        # scalar product
        if isinstance(a, int) or isinstance(a, float):
            res=list(map(lambda x: np.divide(x, a), self.data))
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(a)
            res=list(map(np.divide, self.data, a.data))
        else:
            Exception("Cannot divide Vectorify and %s" % (type(a)))
        return Vectorify(res)

    def dot(self, a):
        if isinstance(a, list):
            a=Vectorify(a)
        assert self.compatible(a)
        prod=self * a
        return sum(list(map(np.sum, prod)))

    def norm(self):
        return sum(list(map(np.sum, (self*self).data)))**0.5

    def assign(self, a):
        if isinstance(a, list):
            a=Vectorify(a)
        if isinstance(a, int) or isinstance(a, float):
            for x in self.data:
                x[:] = a
        # elementwise product
        elif isinstance(a, Vectorify):
            assert self.compatible(a)
            for x,y in zip(self.data, a.data):
                x[:] = y
        else:
            Exception("Cannot assign %s to Vectorify" % (type(a)))

    def copy(self):
        return Vectorify([x.copy() for x in self.data])

    def __getitem__(self, i):
        return self.data[i]

    def save(self, filename):
        pickle.dump(self.data, open(filename, "wb") )
