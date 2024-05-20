def binary_search(array: list, target, compare=None):
    # index of first number in the array
    leftIndex = 0
    # index of last number in the array
    rightIndex = len(array) - 1

    # if compare method is not parased into here
    if compare is None:
        # create a new compare form to check if x is less than y
        compare = lambda x, y: x < y

    while leftIndex <= rightIndex:
        # middle index between left and right
        middleIndex = round((leftIndex + rightIndex) / 2)
        # get the corresponding element in the middle index
        element = array[middleIndex]

        # calculate the difference 
        # between target element and element in the middle index
        diff = compare(target, element)


        # if we can not compare them
        if diff is None:
            # return index in the middle and false
            return middleIndex, False

        # if target "<" element
        if diff:
            # target should be between left and middle
            # set rightIndex to middleIndex - 1
            rightIndex = middleIndex - 1
        # else if target ">" element
        else:
            # target should be between middle and right
            # set leftIndex to middleIndex + 1
            leftIndex = middleIndex + 1
            
    # find the index at which we can insert target and return true
    return leftIndex, True
