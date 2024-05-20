import numpy as np
from numpy.linalg import norm
from scipy.optimize import linprog
from qpsolvers import solve_ls
from filters.pair_filter import PairFilter

class HyperplaneFilter(PairFilter):
    '''
    Xie, Min, Raymond Chi-Wing Wong, and Ashwin Lall.
    "Strongly truthful interactive regret minimization."
    In Proceedings of the 2019 International Conference on Management of Data, pp. 281-298. 2019.

    For every new item x, run the following LP:
    min_u constant
    s.t.
    u^T(x-z) >= eps for all (y,z) in P,
    u^T(z-y) >= 0 for all (y,z) in P.
    If the LP is infeasible, it means that x is worse than at least one item in P.
    '''
    def __init__(self, epsilon: float, compare, max_iter=4000):
        self.epsilon = epsilon  # tol
        self.compare = compare  # ult 
        self.max_iter = max_iter
        self.unComp = []  # C keep uncomparable items
        self.Pair = [] # sorted sample/representatives in ascending order
        self.current_pair = []  # p current pair

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

    def __iter__(self):
        for x,y in self.Pair:
            yield x
            yield y
        yield from self.unComp

    def prune_basic(self, x):
        if_prune = False
        for y in self:  # __iter__
            if np.isclose(norm(x - y), self.epsilon) or norm(x - y) < self.epsilon:
                if_prune = True
        return if_prune

    def prune(self, x):
        if self.prune_basic(x):
            return True

        Pairs, length = self.Pair, len(self.Pair)
        if length == 0:
            return False
            
        # min_x c^Tx s.t. A_ub x <= b_ub
        pairLength = len(self.Pair[0][0])
        c = np.zeros(pairLength)
        diffArray = [dataPoint_b-dataPoint_a for dataPoint_a,dataPoint_b in self.Pair]
        diffArray.extend([(1-self.epsilon)*x - dataPoint_b for dataPoint_a,dataPoint_b in self.Pair])

        diffArray  = -np.array(diffArray )
        fillWithZero = np.zeros(len(self.Pair))
        fillWithOne = -np.ones(len(self.Pair))
        newArray = np.concatenate([fillWithZero, fillWithOne])

        res = linprog(c, A_ub=diffArray , b_ub=newArray, bounds=(None,None), method='highs', options={'maxiter': self.max_iter})

        if res.status == 0: # success
            return False
        if res.status == 2: # infeasible
            return True
        return False # LP fails

    def find_top(self, X):
        best = None
        for x in X:
            if best is None:
                best = x
            if self.compare(best, x): # drop x if uncomparable
                best = x
        return best

    def wrapup(self):
        X = [z for (y,z) in self.Pair]
        X.extend(self.unComp)
        return self.find_top(X)