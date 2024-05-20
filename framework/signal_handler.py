import signal
from functools import partial

def stream_end(framework):
    print(" >>> Output current best:")
    best = framework.output_current_best()
    print(best)

def signal_handler(framework, signal, frame):
    print("\n-------------------------------------------------------------")
    print("Terminating the input stream")
    stream_end(framework)
    exit(0)


def handler_setup(framework):
    # if entering crtl+c to stop the input stream
    # return current best for now 
    # not for the all data points
    partial_signal_handler = partial(signal_handler, framework)
    signal.signal(signal.SIGINT, partial_signal_handler)
