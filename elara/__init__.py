__version__ = "1.0.0"

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_benchmark_data(path):
    return os.path.join(_ROOT, 'example_benchmark_data', path)


class ConfigError(KeyError):
    """
   Configuration error resulting in KeyError.
   """
    pass


class ConfigHandlerError(ConfigError):
    """
    Unrecognised handler name in configuration resulting in KeyError.
    """
    pass


class ConfigInputError(ConfigError):
    """
    Unrecognised input name in handler requirements resulting in KeyError.
    """
    pass


class ConfigPostProcessorError(ConfigError):
    """
    Unrecognised postprocessor name in configuration resulting in KeyError.
    """
    pass


class PostProcessorPrerequisiteError(ConfigError):
    """
    Missing post processor prerequisite handler resulting in ConfigError
    """
    pass


class ConfigBenchmarkError(ConfigError):
    """
    Unrecognised benchmark name in configuration resulting in KeyError.
    """
    pass
