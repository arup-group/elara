__version__ = "1.0.0"

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_benchmark_data(path):
    return os.path.join(_ROOT, 'benchmark_data', path)
