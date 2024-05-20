import numpy as np
# from typing import Callable
from framework.counter import Counter

def lt_wrt_u(utilityVector, totalComparisonCount: Counter=None, failureComparisonCount: Counter=None, gap: float=None):
    # create a inner function wrapper() given data point x and data point y
    def wrapper(dataPoint_x, dataPoint_y):
        # create a Counter() to count the total number of comparison
        # between one data point and another data point
        # only increase the count if Counter() is initialized and given
        if totalComparisonCount is not None:
            # increase the count in the Counter() by 1
            totalComparisonCount.increase()

        # dataPoint_y - dataPoint_x
        # e.g. [4, 5, 6] - [1, 2, 3] = [3, 3, 3]
        # vector & utilityVector
        # e.g [3, 3, 3] @ [1, 1, 1] = 3*1 + 3*1 + 3*1 = 9
        uxy = (dataPoint_y - dataPoint_x) @ utilityVector
        # get the absolute value of uxy
        gxy = np.abs(uxy)


        # if the relative tolerance between uxy and 0 is less than 1e-08
        # 1e-08 is set by default in the np.isclose() function
        if np.isclose(uxy, 0):
            # create a Counter() to count the total number of comparison failed
            # between one data point and another data point
            # only increase the count if Counter() is initialized and given
            if failureComparisonCount is not None:
                # increase the count in the Counter() by 1
                # means that dataPoint_x and dataPoint_y is very close
                # fail to compare
                failureComparisonCount.increase()
            return None

        # check whether gap is set and given 
        if gap is not None:
            # if gxy is less than gap we set or gxy and gap is very close
            if gxy < gap or np.isclose(gxy, gap):
                if failureComparisonCount is not None:
                # increase the count in the Counter() by 1
                # means that dataPoint_x and dataPoint_y is very close
                # fail to compare
                    failureComparisonCount.increase()
                return None

        return uxy > 0
    
    return wrapper