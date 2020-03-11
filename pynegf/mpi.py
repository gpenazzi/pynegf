import logging

_HAS_MPI = False

try:
    import mpi4py
    _HAS_MPI = True
except ModuleNotFoundError:
    logging.info('Module mpi4py not found. MPI support has been disabled.')
    _HAS_MPI = False


def has_mpi():
    """
    Returns:
        bool: whether mpi is supported or not.
    """
    return _HAS_MPI


def get_world_comm():
    """
    Returns the world communicator if mpi support is enabled.
    Otherwise, returns None.
    """
    if _HAS_MPI:
        from mpi4py import MPI
        return MPI.COMM_WORLD

    return None
