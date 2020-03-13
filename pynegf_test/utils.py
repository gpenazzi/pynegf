"""
Module with testing utilities.
"""
import numpy
import scipy


def orthogonal_linear_chain(nsites=100, contact_size=20, coupling=1.0):
    """
    Build the hamiltonian of an orthogonal nearest neighbor
    linear chain with the correct arrangement for the contacts.
    Note that this simple function goes through a dense matrix and
    it is not suitable for very large matrices.

    Args:
        nsites (int): the number of sites.
        coupling (complex): the hopping matrix element.
    """
    if (contact_size >= nsites / 2):
        raise ValueError("Contacts are too large")

    mat = numpy.zeros(shape=(nsites, nsites), dtype='complex128')

    for ii in range(nsites - contact_size):
        mat[ii, ii - 1] = numpy.conj(coupling)
        mat[ii - 1, ii] = coupling
    for ii in range(nsites - contact_size, nsites):
        mat[ii, ii - 1] = numpy.conj(coupling)
        mat[ii - 1, ii] = coupling
    mat[80, 0] = numpy.conj(coupling)
    mat[0, 80] = coupling

    mat_csr = scipy.sparse.csr_matrix(mat)
    mat_csr.sort_indices()

    return mat_csr
