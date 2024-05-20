import numpy as np

from scipy.optimize import linprog # for test_lp()
from qpsolvers import solve_ls # for test_leastsq()
from numpy.random import default_rng # for test_filter()
from numpy.linalg import norm

from framework.compare import lt_wrt_u
from framework.binary_search import binary_search
from framework.counter import Counter
from framework.aux_func import sphere
from filters.list_filter import ListFilter



# test comparison function
def test_lt_wrt_u():
    # normal
    u = np.array([1, 1, 1])
    c = Counter()
    cerr = Counter()
    gap = 0.5
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    lt = lt_wrt_u(u, c, cerr, gap)
    result = lt(x, y)
    assert result == True
    assert c.count == 1
    assert cerr.count == 0

    # equal
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    c = Counter()
    cerr = Counter()
    lt = lt_wrt_u(u, c, cerr, gap)
    result = lt(x, y)
    assert result == None
    assert c.count == 1
    assert cerr.count == 1

    # < gap
    x = np.array([1, 2, 3])
    y = np.array([1.1, 2.1, 3.1])
    c = Counter()
    cerr = Counter()
    lt_fn = lt_wrt_u(u, c, cerr, gap)
    result = lt_fn(x, y)
    assert result == None
    assert c.count == 1
    assert cerr.count == 1

# test binary search
def test_binary_search():
    # normal
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target = np.array([1, 1, 1])
    result = binary_search(array, target, lt_wrt_u(np.array([1, 1, 1])))
    assert result[0] == 0
    assert result[1] == True

    # equal
    target = np.array([4, 5, 6])
    result = binary_search(array, target, lt_wrt_u(np.array([1, 1, 1])))
    assert result[0] == 1
    assert result[1] == False

    # 合成数据单独测试二分查找  
    # no compare
    array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 21]
    assert binary_search(array, 8)[0] == 4
    assert binary_search(array, 0.5)[0] == 0
    assert binary_search(array, 22)[0] == len(array)

# test quadratic programming
def test_leastsq():
    A = np.array([[1., ], [1., ], [1., ]])
    b = np.array([1., 1., 0.])
    G = np.ones((1, 1))
    h = np.array([0])
    sol = solve_ls(A, b, G=-G, h=h, solver='osqp')
    assert np.isclose([0.66666667], sol)

    def lq(S, x):
        s, S, x = len(S), np.array(S), np.array(x)
        R = [S[i - 1] - S[i] for i, _ in enumerate(S) if i > 0]
        R = np.array(R).T
        c = x - S[-1]
        lb = np.zeros(s - 1)  # box ineqs

        est = solve_ls(R, c, lb=lb, solver='osqp')
        return est, norm(R @ est - c)

    S = [
        [0.8, 0],
        [0.91, 0],
    ]
    x = [0.9, 0]
    est, sol = lq(S, x)
    assert est is not None
    assert sol < 0.05


# test linear programming
def test_lp():
    # min_x c^Tx s.t. A_ub x <= b_ub
    c = np.zeros(2)
    A = [
        [1, 0],
        [0, 1],
        [-1, -1],
    ]
    A = np.array(A)
    b = np.array([0, 0, -1])
    res = linprog(c, A_ub=A, b_ub=b, bounds=None, method='highs')

    assert res.status == 2  # infeasible

# test list-filter with no tie
def test_filter():
    rng = default_rng(42)

    n, d, sz = 1000, 10, 30
    D = sphere(n, d, rng)
    u = sphere(1, d, rng).flatten()

    tol = 0.01
    lt = lt_wrt_u(u)
    # ft = Filter(tol, lt)
    ft = ListFilter(tol, lt, param_solver='QP')

    npruned = 0
    best, ubest = None, -1e7
    unpruned = []
    for i in range(len(D)):
        x = D[i]

        # test if x is better than best
        ux = x @ u
        if ux > ubest:
            ubest = ux
            best = x

        if len(ft) < sz:
            ft.add(x)

        # Prune
        pruned = False
        if ft.prune(x):
            npruned = npruned + 1
        else:
            unpruned.append(x)

    # Make sure ft contains at least one feasible item
    print('unpruned:', len(unpruned), 'out of', len(D))
    uunp = max([y @ u for y in unpruned] + [-1e7])
    uft = max([y @ u for y in ft]) + tol
    uft = max(uunp, uft)
    assert uft > ubest or np.isclose(uft, ubest)

# test list-filter with tie
def test_filter_ties():
    u = np.array([1,0])
    D = [
        [0.9, 0],
        [0.91, 0],
        [0.8, 0],
        [0.81, 0],
        [0.5, 0],
        [0.51, 0],
        [0.91, 0.3],
        [0.91, 0.5],
        [0.91, 0.8],
    ]
    D = np.array(D)

    tol, gap = 0.005, 0.05
    ult = lt_wrt_u(u, gap=gap)
    ft = ListFilter(tol, ult, occur_tie=True, param_solver='LP')

    for i in range(len(D)):
        x = D[i]
        if ft.prune(x):
            continue

        ft.add(x)

    assert len(ft.sample) == 3
    assert len(ft.group) == 3
    group = [len(i) for i in ft.group.values()]
    assert min(group) == 1 # 0.8
    assert max(group) == 5 # all 0.91
