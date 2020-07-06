from ctypes import util
import collections
import ctypes
import logging
import numpy
import os
import typing

# Package imports.
from .pynegf import PyNegf
from .version import __version__

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
    def _set_dependencies():
        """
        Set defaults for the list of dependencies linked at runtime.
        This includes libnegf, blas and lapack.
        """
        # Add libnegf. The default install location is in the .lib folder.
        # If it fails, look for a system libnegf.
        path = os.path.dirname(__file__)
        print('DEBUG PRINT ', path, os.listdir(path))
        print('DEBUG PRINT ', os.listdir(os.path.join(path, 'lib')))
        path = util.find_library(os.path.join(path, 'lib/libnegf.so'))
        if path is None:
            try:
                os.environ['LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
                path = util.find_library('negf')
            except KeyError:
                pass
        dependencies = {}
        dependencies['negf'] = {
            'paths': [path]}

        return dependencies

    @staticmethod
    def defaults() -> typing.Dict[str, typing.Any]:
        """
        Returns a dictionary of default settings. The available settings are:

        `negf_path`: path of libnegf
        """
        defaults = {
            'dependencies': Settings._set_dependencies()}
        defaults['loglevel'] = logging.INFO

        return defaults


# Initialize global settings.
settings = Settings()

# Inizialize runtime parameters.
RuntimeEnvironment = collections.namedtuple(
    'RuntimeEnvironment',
    ['dependencies', 'logger'])


def _load_dependencies() -> typing.Dict[str, ctypes.CDLL]:
    """
    Load at module level the ctypes dependencies.

    Returns:
        A dictionary containing the loaded library instances.
    """
    # Load libraries
    error_message = (
        "Resource {0} was not found. Provide a list of paths in "
        " pynegf.settings['dependencies']['negf']['paths'] "
        " and run >>> pynegf.load_runtime_environment().")

    dependencies = {}
    for key, val in [
            x for x in settings['dependencies'].items() if x[0] != 'negf']:
        if not val['paths']:
            logging.warning(error_message.format('key'))
        for path in val['paths']:
            dependencies[key] = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    try:
        dependencies['negf'] = ctypes.CDLL(
            settings['dependencies']['negf']['paths'][0])
    except OSError as err:
        logging.warning(error_message.format('negf'))
        logging.warning(err)

    return dependencies


def load_runtime_environment() -> RuntimeEnvironment:
    """
    Load runtime environment as specified in pynegf.settings.s
    """
    global runtime_environment
    runtime_environment = RuntimeEnvironment(
        dependencies=_load_dependencies(),
        logger=logging.getLogger('pynegfLogger'))
    runtime_environment.logger.setLevel(settings['loglevel'])

    return runtime_environment


runtime_environment = load_runtime_environment()


def cdll_libnegf() -> ctypes.CDLL:
    """
    Return:
        The loaded CDLL object
    """
    return runtime_environment.dependencies['negf']
