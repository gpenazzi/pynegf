from ctypes import util
import collections
import ctypes
import logging
import numpy
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
    def _set_dependencies():
        """
        Set the list of dependencies linked at runtime. This includes
        libnegf, blas and lapack.
        """
        # Add libnegf.
        path = util.find_library('negf')
        dependencies = {}
        dependencies['negf'] = {
            'paths': [path]}

        # Add blas and lapack from numpy. The first found is used.
        numpy_config = numpy.__config__
        print('DEBUG ', numpy.show_config())
        blas_info = [
            x for x in dir(numpy_config) if 'blas' in x and '_info' in x][0]
        if blas_info:
            blas_info = getattr(numpy_config, blas_info)
            dependencies['blas'] = {
                'paths': [
                    util.find_library(x) for x in blas_info['libraries']]}
        # Fallback to sistem blas is any.
        else:
            dependencies['blas'] = {
                'paths': [util.find_library('blas')]}

        lapack_info = [
            x for x in dir(numpy_config) if 'lapack' in x and '_info' in x][0]
        if lapack_info:
            lapack_info = getattr(numpy_config, lapack_info)
            dependencies['lapack'] = {
                'paths': [
                    util.find_library(x) for x in lapack_info['libraries']]}
        # Fallback to sistem lapack is any.
        else:
            dependencies['lapack'] = {
                'paths': [util.find_library('lapack')]}

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
    global _dependencies
    # Load libraries
    error_message = (
        "Resource {0} was not found. Provide a list of paths in "
        " pynegf.settings['dependencies']['paths'] "
        " and run >>> pynegf.load_runtime_environment().")

    dependencies = {}
    try:
        dependencies['negf'] = ctypes.CDLL(
            settings['dependencies']['negf']['paths'][0])
    except OSError:
        logging.warning(error_message.format('negf'))

    for key, val in [
            x for x in settings['dependencies'].items() if x[0] != 'negf']:
        if not val['paths']:
            logging.warning(error_message.format('key'))
        for path in val['paths']:
            dependencies[key] = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

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
