import ctypes
import ctypes.util
import logging
import os

# Internal imports.
from .pynegf import PyNegf

# The CDLL library object. We define it at module level. 
_CDLL_LIBNEGF = None

def cdll_libnegf():
    """
    Return:
        The loaded CDLL object
    """
    return _CDLL_LIBNEGF


def load_libnegf(library_path=None):
    """
    Load at module level the libnegf library. 

    Args:
        library_path (string): the full path to the library, e.g. `/usr/lib64/libnegf.so`
        If no path is given, ctypes.utils.find_library and some default paths are used.  
    
    Returns:
        The ctypes.CDLL library object.

    Raises:
        OSError: if the library could not be loaded. 
    """
    cdll_object = None
    # Look for the library if no path was given. 
    if library_path is None:
        lib = ctypes.util.find_library('negf')
        if lib is not None:
            # Manually loop over some usual paths.
            # TODO: hopefully find_library will give a full path one day and we won't need to do this. 
            for root in ['/usr/lib', '/usr/lib64', '/usr/local/lib']:   
                try:
                    # mode=1 is lazy loading, to avoid some troubles with unused
                    # BLAS symbols in sparsekit. 
                    cdll_object = ctypes.CDLL(os.path.join(root, lib), mode=1)        
                except OSError:
                    pass
        # If we are not succesful raise, otherwise return.  
        if cdll_object is None:
            raise OSError("libnegf not found in any default path.")
        else:
            return cdll_object
    # Otherwise look in the user given path. 
    else:
        return ctypes.CDLL(library_path, mode=1)

# Define the default logger.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Try to load the library using the default paths
try:
    _CDLL_LIBNEGF = load_libnegf()
except OSError:
    logging.warning('libnegf.so not found. The library must be loaded '
                    'specifying a path in pynegf.load_library().')
