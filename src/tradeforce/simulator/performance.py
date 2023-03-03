"""_summary_

Returns:
    _type_: _description_
"""

import time
import numpy as np
from numba import njit, objmode  # type: ignore

NB_CACHE = False


class Performance:
    """_summary_"""

    def __init__(self):
        self.results = {}
        self.tree = {"stack": ["main"], "main": set()}

    def wrapper_objm_start(self, f):
        start = time.time()
        self.tree[self.tree["stack"][-1]].add(f.__name__)
        self.tree["stack"] += [f.__name__]
        if f.__name__ not in self.results:
            self.tree[f.__name__] = set()
            # print(tree['stack'])
        return start

    def wrapper_objm_end(self, f, start):
        run_time = time.time() - start
        if f.__name__ in self.results:
            self.results[f.__name__] += [run_time]
        else:
            self.results[f.__name__] = [run_time]
        self.tree["stack"] = self.tree["stack"][:-1]

    def t(self, f):
        def wrapper(self, *args, **kwargs):
            start = self.wrapper_objm_start(f)
            g = f(*args)
            self.wrapper_objm_end(f, start)
            return g

        return wrapper

    def jit_timer(self, f):
        jf = njit(cache=NB_CACHE, parallel=False)(f)

        @njit(cache=NB_CACHE, parallel=False)
        def wrapper(*args):
            with objmode(start="float64"):
                start = self.wrapper_objm_start(f)
            g = jf(*args)
            with objmode():
                self.wrapper_objm_end(f, start)
            return g

        return wrapper

    def get_tree(self):
        return self.tree

    def get_results(self):
        return self.results

    def print_tree(self, nodes, layer):
        for node in nodes:
            print("{:.6f}".format(np.min(self.results[node])), "-|-" * layer, node)
            self.print_tree(self.tree[node], layer + 1)
