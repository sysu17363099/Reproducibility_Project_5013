import numpy as np
from numpy.linalg import norm
from scipy.optimize import linprog
from qpsolvers import solve_ls


class PairFilter:
    '''
    Filter by a cone formed by pairs of items
    Citation:
    The pruning algorithm implemented in this filter from the paper by Xie et al.

    Original Paper: Min Xie, Raymond Chi-Wing Wong, and Ashwin Lall. 2019. Strongly truthful
    interactive regret minimization. In Proceedings of the 2019 International Conference
    on Management of Data. 281–298.

    Code implemented by Guangyi Zhang, Nikolaj Tatti and Aristides Gionis

    Code: https://github.com/Guangyi-Zhang/interactive-favourite-tuples/blob/main/irm/filter.py
    '''

    def __init__(self, epsilon: float, compare, param_solver='QP', epsilon_c=1.0, max_iter=4000):
        self.epsilon = epsilon  # tol
        self.compare = compare  # ult
        self.param_solver = param_solver
        self.epsilon_c = epsilon_c # true_u if true_u < 1, we set tol = tol*true_u in pruning    
        self.max_iter = max_iter
        self.unComp = []  # C keep uncomparable items
        self.Pair = []  # P list of pairs, given as (x,y) s.t. x < y
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
    
    def __len__(self):
        return len(self.Pair) * 2 + len(self.current_pair) + len(self.unComp)

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

    def prune_lp(self, x) -> bool:
        # min_x c^Tx s.t. A_ub x <= b_ub, A_eq x = b_eq
        if_prune = False
        R1 = list()
        R2 = list()
        Pairs, length = self.Pair, len(self.Pair)
        for (y,z) in Pairs:
            R1.append(y-z)
            R2.append(z)

        R = np.concatenate([R1,R2])
        R = np.transpose(np.array(R))

        A = np.array([0]*length + [1]*length).reshape((1,-1))
        b = np.array([1])

        A_eq = np.concatenate([R,A])
        b_eq = np.concatenate([x,b])
        c = np.zeros(A_eq.shape[1])

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs', options={'maxiter': self.max_iter})

        if res.status == 0: # success
            if_prune = True
        return if_prune

    def prune(self, x):
        if self.prune_basic(x):
            return True

        Pairs, length = self.Pair, len(self.Pair)
        if length == 0:
            return False

        if self.param_solver == 'LP':
            return self.prune_lp(x)

        # Run constrained least-squares below
        # min_x norm(Rx-c) s.t. Gx <= h and Ax = b
        R1 = list()
        R2 = list()
        for (y, z) in Pairs:
            R1.append(y - z)
            R2.append(z)

        R = np.concatenate([R1,R2])
        R = np.array(R).T
        c = x
        lb = np.zeros(2*length) # box ineqs

        A = np.array([0]*length + [1]*length)
        b = np.array([1])
        est = solve_ls(R, c, A=A, b=b, lb=lb, max_iter=self.max_iter, solver='osqp')

        if est is None: # fails to solve
            return False
        return np.isclose(norm(R @ est - c), self.epsilon * self.epsilon_c) or norm(R @ est - c) < self.epsilon * self.epsilon_c

    def find_top(self, X):
        xbest = None
        for x in X:
            if xbest is None:
                xbest = x
            else:
                if self.compare(xbest, x):
                    xbest = x
                    
        return xbest

    def output_current_best(self):
        X = [z for (y,z) in self.Pair]
        X.extend(self.unComp)
        return self.find_top(X)