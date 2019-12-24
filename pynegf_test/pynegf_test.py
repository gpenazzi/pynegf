import numpy
import pynegf
import pytest
import scipy

# Skip if libnegf is not available.
if pynegf.cdll_libnegf() is None:
    pytest.skip(
        "libnegf backengine not available on the system",
        allow_module_level=True)


def test_construction():
    """ Test that we can construct a NEGF instance, without performing any operation. """
    foo = pynegf.PyNegf()
    assert(foo is not None)

def test_transmission_linear_chain():
    """ Test that we can calculate the transmission for an ideal linear chain. """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat = numpy.zeros(shape=(100,100), dtype='complex128')
    for ii in range(80):
        mat[ii, ii - 1] = 1.0
        mat[ii - 1, ii] = 1.0
    for ii in range(81, 100):
        mat[ii, ii - 1] = 1.0
        mat[ii - 1, ii] = 1.0
    mat[0, 80] = 1.0
    mat[80, 0] = 1.0

    mat_csr = scipy.sparse.csr_matrix(mat)
    mat_csr.sort_indices()
    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(100)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array([79, 99]),
        numpy.array([59, 79]),
        numpy.array([14, 29, 44, 59]),
        numpy.array([3, 0]))

    # Initialize parameters relevant for the transmission.
    negf.params.emin = -3.0
    negf.params.emax = 3.0
    negf.params.estep = 0.01
    negf.params.mu[0] = 0.1
    negf.set_params()
    negf.print_tnegf()

    # Set also some local DOS intervals.
    negf.set_ldos_intervals(numpy.array([0, 30, 0]), numpy.array([59, 59, 29]))
    negf.solve_landauer()

    #Get transmission, dos and energies as numpy object
    energies = negf.energies()
    print('energies', energies)
    trans = negf.transmission()
    ldos = negf.ldos()
    currents = negf.currents()
    print('Currents',currents)
    print('trans', trans)
