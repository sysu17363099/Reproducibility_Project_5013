import numpy as np
from numpy.linalg import norm
from itertools import product
from scipy.optimize import linprog
from qpsolvers import solve_ls
from framework.binary_search import binary_search


class ListFilter:
    def __init__(self, epsilon, compare, param_solver='QP', occur_tie=False, epsilon_c=1.0, max_iter=4000):
        self.epsilon = epsilon # tol
        self.compare = compare # ult
        self.param_solver = param_solver # solver
        self.occur_tie = occur_tie # ties
        self.epsilon_c = epsilon_c # true_u
        self.sample = list() # S
        self.group = dict() # Gs
        self.max_iter = max_iter

    def add(self, target):
        index, is_found = binary_search(self.sample, target, compare=self.compare)
        # no tie
        if is_found:
            self.sample.insert(index, target)
            if self.occur_tie:
                self.group[hash(target.data.tobytes())] = [target]
        # occur tie
        else:
            if not self.occur_tie:
                return # discard the target value

            # insert target value into the group with corresponding index
            insert_value = self.sample[index]
            
            groups = self.group[hash(insert_value.data.tobytes())]
            groups.append(target)

    def __len__(self):
        if self.occur_tie == False:
            length = len(self.sample)
            return length
        else:
            if not len(self.group) == 0:
                total = 0
                for G in self.group.values():
                    total += len(G)
            elif len(self.group) == 0:
                return 0

    def __iter__(self):
        if self.occur_tie == False:
            for value in self.sample:
                yield value
        else:
            for G in self.group.values():
                for value in G:
                    yield value

    # prune_lp
    def prune_by_lp(self, x):
        if_prune = False
        sample = self.sample
        filter_len = len(self.sample)
        neigh_different = list()

        for i in range(1, filter_len):
            neigh_different.append(sample[i - 1] - sample[i])

        neigh_different = np.transpose(np.array(neigh_different))
        best_different = x - sample[filter_len-1]
        func_ratio = np.zeros(neigh_different.shape[1])

        output = linprog(func_ratio, A_eq=neigh_different, b_eq=best_different, bounds=(0, None), method='highs', options={'maxiter': self.max_iter})

        if output.status == 0:  
            if_prune = True
        return if_prune

    # prune_cone
    def prune_by_qp(self, x):
        sample = self.sample
        s = len(self.sample)

        R = []
        for i, _ in enumerate(sample):
            if i > 0:
                R.append(sample[i - 1] - sample[i])
        R = np.array(R).T
        c = x - sample[-1]
        lb = np.zeros(s - 1)

        est = solve_ls(R, c, lb=lb, max_iter=self.max_iter, solver='osqp')

        if est is not None:
            distance = norm(R @ est - c)
            tolerance = self.epsilon * self.epsilon_c
            if np.isclose(distance, tolerance):
                return True
            elif distance < tolerance:
                return True
            else:
                return False
        else:
            return False

    # prune_lp_ties
    def prune_lp_with_ties(self, x):
        if_prune = False
        neigh_different = list() # R1
        best_different = list() # R2
        groups = list() # Gs
        best_group = list() # Gbest
        final_group = list() # R

        for y in self.sample:
            groups.append(self.group[hash(y.data.tobytes())])

        for group0, group1 in zip(groups, groups[2:]):
            for y0, y1 in product(group0, group1):
                different = y0-y1
                neigh_different.append(different)

        for i in [len(self.sample)-1, len(self.sample)-2]:
            # best_group.append(self.group[key(self.sample[i])])
            best_group = self.group[hash(self.sample[i].data.tobytes())]
            for y in best_group:
                best_different.append(y)

        # if len(groups) == 0:
        if len(neigh_different) == 0:
            final_group = best_different
        else:
            # final_group = np.vstack((neigh_different, best_different))
            final_group = np.concatenate([neigh_different, best_different])
        final_group = np.transpose(np.array(final_group))

        # A = np.vstack((np.zeros(len(neigh_different)), np.ones(len(best_different)))).reshape((1, -1))
        A = np.array([0]*len(neigh_different) + [1]*len(best_different)).reshape((1,-1))
        b = np.array([1])

        # A_eq = np.vstack((final_group, A))
        A_eq = np.concatenate([final_group,A])
        # b_eq = np.vstack((x, b))
        b_eq = np.concatenate([x,b])
        coefficients = np.zeros(A_eq.shape[1])

        output = linprog(coefficients, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs', options={'maxiter': 4000})

        if output.status == 0:
            if_prune = True
        return if_prune

    # prune_cone_ties
    def prune_qp_with_ties(self, x):
        R = []
        group = []
        for y in self.sample:
            group.append(self.group[hash(y.data.tobytes())])

        for group1, group2 in zip(group, group[2:]):
            for y1 in group1:
                for y2 in group2:
                    R.append(y1-y2)
        # for i in range(len(group) - 1):
        #     G1 = group[i]
        #     G2 = group[i + 2]
        #     for j in range(len(G1)):
        #         for k in range(len(G2)):
        #             R.append(G1[j] - G2[k])
        nR1 = len(R)

        for idx in [len(self.sample)-2, len(self.sample)-1]:
            group_best = self.group[hash(self.sample[idx].data.tobytes())]
            for y in group_best:
                R.append(y)

        nR = len(R)
        R = np.transpose(np.array(R))
        c = x
        lb = np.zeros(nR)

        A = np.array([])
        for _ in range(nR1):
            A = np.append(A, 0)
        for _ in range(nR - nR1):
            A = np.append(A, 1)
        b = np.array([1])

        est = solve_ls(R, c, A=A, b=b, lb=lb, max_iter=self.max_iter, solver='osqp')

        if est is not None:
            distance = norm(R @ est - c)
            tolerance = self.epsilon * self.epsilon_c
            if np.isclose(distance, tolerance):
                return True
            elif distance < tolerance:
                return True
            else:
                return False
        else:
            return False

    def prune(self, target):
        if self.basic_prune(target):
            return True
        elif len(self.sample) < 2:
            return False     
        elif self.occur_tie:
            if self.param_solver == 'LP':
                return self.prune_lp_with_ties(target)
            return self.prune_qp_with_ties(target)
        else:
            if self.param_solver == 'LP':
                return self.prune_by_lp(target)
            return self.prune_by_qp(target)
    
    def basic_prune(self, target) -> bool:
        for value in self: # __iter__
            norm_vec = norm(target - value) # np.norm
            less_than = np.isclose(norm_vec, self.epsilon) or norm_vec < self.epsilon
            if less_than:
                return True
        return False

    # wrapup
    def output_current_best(self):
        length = len(self.sample)
        # print("sample", length)
        return self.sample[length-1]
