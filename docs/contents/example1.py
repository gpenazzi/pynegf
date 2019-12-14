import numpy
import pynegf
import scipy


def transmission_linear_chain():
    """
    Calculate the transmission for a linear chain model hamiltonian.
    """
    # Start an instance of the library.
    negf = pynegf.PyNegf()

    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat = numpy.zeros(shape=(100, 100), dtype='complex128')
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

    # Pass the hamiltonian to libnegf.
    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(100)

    # Initialize the system structure. Here we specify the following
    # parameters:
    #   number of contact: 2
    #   start-end index of first contact: numpy.array([80, 100])
    #   start-end index of second contact: numpy.array([60, 80])
    #   end-index of each layer, for the iterative algorithm: numpy.array([15, 30, 45, 60])
    #   index of bloks interacting with the contact: numpy.array([4, 1])
    negf.init_structure(
        2,
        numpy.array([80, 100]),
        numpy.array([60, 80]),
        numpy.array([15, 30, 45, 60]),
        numpy.array([4, 1]))

    # Initialize parameters relevant for the transmission.
    # the chemical potential mu is used for evaluating the current only.
    negf.params.emin = -3.0
    negf.params.emax = 3.0
    negf.params.estep = 0.01
    negf.params.mu[0] = 0.1
    negf.set_params()
    negf.print_tnegf()

    # Set also some local DOS intervals.
    negf.set_ldos_intervals(numpy.array([1, 31, 1]), numpy.array([60, 60, 30]))
    negf.solve_landauer()

    # Get transmission, dos and energies as numpy object.
    energies = negf.energies()
    print('energies', energies)
    trans = negf.transmission()
    ldos = negf.ldos()
    currents = negf.currents()
    print('Currents',currents)
    print('trans', trans)