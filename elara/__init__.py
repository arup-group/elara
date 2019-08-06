__version__ = "1.0.0"

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_benchmark_data(path):
    return os.path.join(_ROOT, 'benchmark_data', path)


class HandlerConfigError(KeyError):
    """
    Incorrect handler name configuration resulting in KeyError.
    """
    pass


class HandlerInputError(KeyError):
    """
    Unknown input requirement resulting in KeyError.
    """
    pass


class ConfigInputError(KeyError):
    """
    Config has found unknown input path requirement resulting in KeyError.
    """
    pass
