class Counter:
    def __init__(self):
        # initialize the count to 0
        self.count = 0

    def increase(self):
        # increase the count by 1
        self.count += 1

    def cur_count(self):
        # return the count in the class counter
        return self.count