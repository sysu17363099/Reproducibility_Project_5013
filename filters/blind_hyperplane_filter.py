import numpy as np
from numpy.linalg import norm
from filters.pair_filter import PairFilter

class BlindHyperplaneFilter(PairFilter):
    '''
    Filter by hyperplanes, each formed by a pair of items.
    It doesn't not guarantee to find a feasible item.
    '''
    def __init__(self, epsilon: float, compare, max_iter=4000):
        self.epsilon = epsilon  # tol
        self.compare = compare  # ult
        self.Pair = []
        self.unComp = []
        self.current_pair = []

    # def __len__(self):
    #     return len(self.Pair) * 2 + len(self.Pair) + len(self.C)

    def __iter__(self):
        for x,y in self.Pair:
            yield x
            yield y
        yield from self.unComp

    def add(self, x):
        self.current_pair.append(x)
        if len(self.current_pair) == 2:
            x, y = self.current_pair
            b = self.compare(x,y)
            if b is None: # ties, keep either one
                self.unComp.append(x)
            else:
                if b:
                    self.Pair.append((x, y))
                else:
                    self.Pair.append((y, x))
            self.current_pair = []

    def prune_basic(self, x) -> bool:
        if_prune = False
        for y in self:  # __iter__
            if np.isclose(norm(x - y), self.epsilon) or norm(x - y) < self.epsilon:
                if_prune = True
        return if_prune
    
    def gt(self, a, b):
        return not np.isclose(a,b) and a > b
    
    def prune(self, x) -> bool:
        '''
        A hyperplane by pair (x,y) s.t. x < y is as follows.
        u^T(x-y) < 0, so any item z s.t. z^T(x-y) > 0 is pruned.
        '''
        if self.prune_basic(x):
            return True

        P = self.Pair
        if len(P) == 0:
            return False

        # k = [gt(x @ (y-z), 0) for (y,z) in P]
        k = []
        for (y,z) in P:
            a = x @ (y-z)
            b = 0
            c = self.gt(a, b)
            k.append(b)
        return np.any(k)