import numpy
import pytest
import time

import pynegf
from pynegf_test import utils

# Skip if libnegf is not available.
if pynegf.cdll_libnegf() is None:
    pytest.skip(
        "libnegf backengine not available on the system",
        allow_module_level=True)


def transmission_2d_orthogonal(nblocks, block_size):
    """
    Calculate the transmission on 100 energy points for a 2d system.
    """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_square_2d_lattice(
        nblocks=nblocks,
        block_size=block_size,
        n_contact_blocks=2,
        coupling=1.0)

    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(nblocks * block_size)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array(
            [(nblocks - 2) * block_size - 1, nblocks * block_size - 1]),
        numpy.array(
            [(nblocks - 4) * block_size - 1, (nblocks - 2) * block_size - 1]))

    # Initialize parameters relevant for the transmission.
    negf.params.g_spin = 1
    negf.params.emin = -5.0
    negf.params.emax = 5.0
    negf.params.estep = 0.1
    negf.params.mu[0] = 0.1
    negf.params.mu[0] = -0.1
    negf.set_params()

    negf.solve_landauer()

    # Get transmission, dos and energies as numpy object
    negf.transmission()


def time_transmission_2d_orthogonal():
    """
    Run a benchmark for various block/block size combination
    """
    for nblocks, block_size in [
            (60, 40), (60, 80), (60, 120)]:
        pynegf.log().info(
            'Running nblocks {} and block_size {}'.format(
                nblocks, block_size))
        start = time.time()
        transmission_2d_orthogonal(nblocks=nblocks, block_size=block_size)
        elapsed_time = time.time() - start
        pynegf.log().info(
            'Elapsed time for nblocks {} and block_size {}: {} s'.format(
                nblocks, block_size, elapsed_time))


if __name__ == "__main__":
    time_transmission_2d_orthogonal()
