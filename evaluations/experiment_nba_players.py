import sys
import pickle
from time import process_time
from numpy.random import default_rng
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur_path))
root_path = '/home/xzhang/zxw/good/code_5009'
print("root_path:", root_path)

sys.path.append(root_path)

from framework.general_framework import GeneralFramework
from filters.list_filter import ListFilter
from framework.aux_func import sphere
from framework.counter import Counter
from framework.compare import lt_wrt_u
from framework.signal_handler import handler_setup


def experiment():
    dataset_path = os.path.join(root_path, "datasets/nba_players.pkl")
    print("This is dataset_path:", dataset_path)

    # compare funtion parameters
    rng = default_rng(42)
    gap = 0.001

    # framework hyperparameters
    epsilon = 0.1
    testbed_threshold = 100
    testbed_theta = 0.5

    # filter hyperparameters
    filter_param = dict()
    filter_param["type"] = "ListFilter"
    filter_param["param_solver"] = "LP"
    filter_param["occur_tie"] = False
    filter_param["epsilon_c"] = 1.0
    filter_param['max_iter'] = 4000

    with open(dataset_path, 'rb') as fin:
        items, names = pickle.load(fin)
        dataset = items.values
        print('Dataset: ', dataset.shape)
        rng.shuffle(dataset)  # in place

    utilityVector = sphere(1, dataset.shape[1], rng).flatten()
    totalComparisonCount, failureComparisonCount = Counter(), Counter()
    compare = lt_wrt_u(utilityVector, totalComparisonCount, failureComparisonCount, gap=gap)
    framework = GeneralFramework(epsilon, compare, testbed_threshold, testbed_theta, filter_param)

    # find best item
    # print("dataset:", dataset)
    xbest, ubest = None, -1e7
    for x in dataset:
        x = x.astype(int)
        ux = utilityVector @ x
        if ux > ubest:
            xbest = x
            ubest = ux
    print('xbest', xbest)
    print('ubest', ubest)

    handler_setup(framework)

    # EXPERIMENT START
    t1_start = process_time()

    for i in range(len(dataset)):
        # print("current#:", i)
        # print("current data:", dataset[i])
        x = dataset[i]
        if framework.prune_by_filters(x):
            continue
        framework.process_new_tuple(x)

    xret = framework.output_current_best()
    t1_stop = process_time()
    # EXPERIMENT END

    print('uret', utilityVector @ xret)

    print('ncmp', totalComparisonCount.cur_count())
    print('ncmperr', failureComparisonCount.cur_count())
    print('mem', len(framework))
    print('runtime', t1_stop - t1_start)


if __name__ == '__main__':
    experiment()