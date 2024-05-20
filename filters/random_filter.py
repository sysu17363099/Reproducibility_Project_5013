import numpy as np
from numpy.linalg import norm


class RandomFilter:
    def __init__(self, epsilon: float, compare, epsilon_c=1.0, max_iter=4000):
        self.epsilon = epsilon  # tol
        self.compare = compare  # ult
        self.epsilon_c = epsilon_c # true_u if true_u < 1, we set tol = tol*true_u in pruning    
        self.max_iter = max_iter
        self.unComp = []  # C keep uncomparable items
        self.sortedSample = [] # sorted sample/representatives in ascending order

    def __len__(self):
        return len(self.sortedSample)

    def __iter__(self):
        yield from self.sortedSample

    def prune_basic(self, x) -> bool:
        if_prune = False
        for y in self:  # __iter__
            if np.isclose(norm(x - y), self.epsilon) or norm(x - y) < self.epsilon:
                if_prune = True
        return if_prune
    
    def prune(self, x) -> bool:
        if self.prune_basic(x):
            return True
        
        return False

    def add(self, x):
        if len(self.sortedSample) >= self.max_iter:
            return

        self.sortedSample.append(x)

    def find_top(self, X):
        xbest = None
        for x in X:
            if xbest is None:
                xbest = x
            else:
                if self.compare(xbest, x):
                    xbest = x
        return xbest
    
    def wrapup(self):
        return self.find_top(self.S)