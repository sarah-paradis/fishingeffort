import timeit
from contextlib import contextmanager
from datetime import datetime


@contextmanager
def _print_with_time(*s):
    """
    Function to calculate the time it takes for a process to complete.
    To use, type "with print_with_time():"
    You should add in parenthesis the action you're doing as a string.
    The function prints out the action and the time it took to complete it in seconds
    """
    print(*s, end=f' [{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] ', flush=True)
    start = timeit.default_timer()
    yield
    print("\t[%.2fs]" % (timeit.default_timer() - start), flush=True)
