from filters.list_filter import ListFilter
from filters.pair_filter import PairFilter
from filters.hyperplane_filter import HyperplaneFilter
from filters.blind_hyperplane_filter import BlindHyperplaneFilter
from filters.random_filter import RandomFilter

class GeneralFramework:
    def __init__(self, epsilon, compare, testbed_threshold, testbed_theta, filter_param: dict):
        # tolerance,nepsilon ε
        self.epsilon = epsilon
        # utility compare function
        self.compare = compare
        # store all the tuples compared
        self.filters = list()
        # store all the tuples compared
        # if the length of this list is not greater than the testbed_threshold we set
        self.testbed = list()
        self.testbed_pruned = set()
        # the maximun length of testbed list
        self.testbed_threshold = testbed_threshold
        self.testbed_theta = testbed_theta
        # arguments in the filter_parm
        # filter_param["type"]: type of filter
        # filter_param["param_solver"]: the type of solver: LP or QP
        # filter_param["occur_tie"]: true/false to indicate if there occurs tie
        # filter_param["epsilon_c"]: current tolerance
        # filter_param['max_iter']: the maximum number of iteration times
        self.filter_param = filter_param
        # parse all the arguments in filter_param into self.new_filter
        self.filter = self.new_filter(filter_param)


    def __len__(self):
        length = 0
        for tuple in self.filters:
            length += len(tuple)
        return length

    def new_filter(self, filter_param):
        newFilter = None
        
        if filter_param['type'] == "ListFilter":
            newFilter = ListFilter(self.epsilon, self.compare, filter_param['param_solver'],
                                filter_param['occur_tie'], filter_param['epsilon_c'], filter_param['max_iter'])
            
        elif filter_param['type'] == "HyperPlaneFilter":
            newFilter = HyperplaneFilter(self.epsilon, self.compare, filter_param['max_iter'])

        elif filter_param['type'] == "BlindHyperPlaneFilter":
            newFilter = BlindHyperplaneFilter(self.epsilon, self.compare, filter_param['max_iter'])
            
        elif filter_param['type'] == "PairHyperPlaneFilter":
            newFilter = PairFilter(self.epsilon, self.compare, filter_param['param_solver'], filter_param['epsilon_c'],
                                filter_param['max_iter'])
            
        elif filter_param['type'] == "RandomFilter":
            newFilter = RandomFilter(self.epsilon, self.compare, filter_param['param_solver'], filter_param['epsilon_c'],
                                filter_param['max_iter'])

        return newFilter

    def find_top(self, X):
        xbest = None
        for x in X:
            if xbest is None:
                xbest = x
            else:
                if self.compare(xbest, x):
                    xbest = x
        return xbest

    def prune_by_filters(self, x):
        if_prune = False
        for filter in self.filters:
            if filter.prune(x):
                if_prune = True
        return if_prune

    def process_new_tuple(self, x):
        if len(self.testbed) < self.testbed_threshold:
            self.testbed.append(x)
            return

        self.filter.add(x)

        cur_pruned = []
        for i, y in enumerate(self.testbed):
            if i not in self.testbed_pruned and self.filter.prune(y):
                cur_pruned.append(i)
        self.testbed_pruned.update(cur_pruned)

        # for y in self.testbed:
        #     if self.filter.prune(y) and y not in self.testbed_pruned:
        #         self.testbed_pruned.append(y)

        if len(self.testbed_pruned) >= self.testbed_theta * self.testbed_threshold:
            self.filters.append(self.filter)
            self.filter = self.new_filter(self.filter_param)
            new_testbed = []
            for idx, data in enumerate(self.testbed):
                if idx not in self.testbed_pruned:
                    new_testbed.append(data)
            self.testbed = new_testbed.copy()
            self.testbed_pruned = set()

    def output_current_best(self):
        self.filters.append(self.filter)
        X = list()
        for filter in self.filters:
            X.append(filter.output_current_best())
        for p in self.testbed:
            X.append(p)
        return self.find_top(X)
