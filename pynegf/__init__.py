import ctypes
import ctypes.util
import logging
import typing


class Settings(dict):
    """
    A global setting dictionary.
    """
    def __init__(self) -> None:
        """
        Construct the settings dictionary.
        """
        # Populate first with all defaults.
        self.update(self.defaults())


    @staticmethod
    def defaults():
        """
        Returns a dictionary of default settings.
        """
        defaults = {}
        # Populate the default paths.
        for lib in ('negf', 'blas', 'lapack'):
            defaults[lib + '_path'] = ctypes.util.find_library(lib)

        return defaults


# Initialize global settings.
settings = Settings()


def load_dependencies() -> typing.Dict[str, ctypes.CDLL]:
    """
    Load at module level the ctypes dependencies.

    Returns:
        A dictionary containing the loaded library instances.
    """

    # Load libraries
    error_message = ("lib{0} was not found. Provide a path in pynegf.settings['{0}_path'] "
                     " and run >>>pynegf.dependencies = pynegf.load_dependencies()")

    try:
        blas_cdll = ctypes.CDLL(settings['blas_path'], mode=ctypes.RTLD_GLOBAL)
    except OSError:
        blas_cdll = None
        logging.warning(error_message.format('blas'))

    try:
        lapack_cdll = ctypes.CDLL(settings['lapack_path'], mode=ctypes.RTLD_GLOBAL)
    except OSError:
        lapack_cdll = None
        logging.warning(error_message.format('lapack'))

    try:
        negf_cdll = ctypes.CDLL(settings['negf_path'])
    except OSError:
        negf_cdll = None
        logging.warning(error_message.format('negf'))

    dependencies = {'negf': negf_cdll, 'blas': blas_cdll, 'lapack': lapack_cdll}

    return dependencies


dependencies = load_dependencies()


def cdll_libnegf() ->  ctypes.CDLL:
    """
    Return:
        The loaded CDLL object
    """
    return dependencies['negf']


# Package imports.
from .pynegf import PyNegf