from ctypes import util
import collections
import ctypes
import logging
import numpy
import os
import typing

# Package imports.
from .pynegf import PyNegf


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
    def _library_path(names, directories):
        """
        Utility to return a path ctypes.util.find_library can find.
        """
        paths = []
        for name in names:
            path = None
            if util.find_library(name):
                return [util.find_library(name)]
            else:
                for directory in directories:
                    path = util.find_library(
                        os.path.join(directory, 'lib' + name + '.so'))
                    if path:
                        paths.append(path)

        return paths


    @staticmethod
    def _set_dependencies():
        """
        Set defaults for the list of dependencies linked at runtime.
        This includes libnegf, blas and lapack.
        """
        # Add libnegf.
        path = util.find_library('negf')
        dependencies = {}
        dependencies['negf'] = {
            'paths': [path]}

        # Add blas and lapack from numpy. The first good one is used.
        numpy_config = numpy.__config__
        blas_info = [
            getattr(numpy_config, x)
            for x in dir(numpy_config) if 'blas' in x and '_info' in x]
        paths = None
        for info in [x for x in blas_info if x]:
            paths = Settings._library_path(
                info['libraries'], info['library_dirs'])
            if paths:
                dependencies['blas'] = {'paths': paths}

        lapack_info = [
            getattr(numpy_config, x)
            for x in dir(numpy_config) if 'lapack' in x and '_info' in x]
        paths = None
        for info in [x for x in lapack_info if x]:
            paths = Settings._library_path(
                info['libraries'], info['library_dirs'])
            if paths:
                dependencies['lapack'] = {'paths': paths}

        if 'blas' not in dependencies.keys():
            paths = Settings._library_path(['blas'], ['', 'usr/local/lib'])
            if paths:
                dependencies['blas'] = {'paths': paths}
        if 'lapack' not in dependencies.keys():
            paths = Settings._library_path(['lapack'], ['', 'usr/local/lib'])
            if paths:
                dependencies['lapack'] = {'paths': paths}

        return dependencies

    @staticmethod
    def defaults() -> typing.Dict[str, typing.Any]:
        """
        Returns a dictionary of default settings. The available settings are:

        `negf_path`: path of libnegf
        `blas_path`: path of libblas
        `lapack_path`: path of liblapack
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
