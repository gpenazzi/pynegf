"""
Module with testing utilities.
"""
import numpy
from scipy import sparse


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
    if contact_size >= nsites / 2:
        raise ValueError("Contacts are too large")
    if contact_size < 2 and contact_size % 2 != 0:
        raise ValueError("Contacts must have 2 principal layers or multiples.")

    mat = numpy.zeros(shape=(nsites, nsites), dtype='complex128')

    for i in range(nsites - contact_size):
        mat[i - 1, i] = coupling
    for i in range(nsites - contact_size, nsites):
        mat[i - 1, i] = coupling
    mat[0, nsites - contact_size] = coupling

    mat_csr = sparse.csr_matrix(mat)
    mat_csr = mat_csr + mat_csr.conjugate(copy=True).transpose()
    mat_csr.sort_indices()

    return mat_csr


def orthogonal_square_2d_lattice(
        nblocks=10,
        block_size=10,
        n_contact_blocks=1,
        coupling=1.0):
    """
    Build a nearest neighbor hamiltonian for a 2d square lattice with contacts
    properly arranged. The modeled lattice is:

    ...*--*--*...
       |  |  |
    ...*--*--*...
       |  |  |
    ...*--*--*...

    The resulting block hamiltonian is:

    0 t   t
    t 0 t   t
      t 0     t
    t     0 t
      t   t 0 t
        t   t 0
    """
    if n_contact_blocks < 2 and n_contact_blocks % 2 != 0:
        raise ValueError("Contacts must have 2 principal layers or multiples.")
    shape = (block_size, block_size)
    onsite_block = numpy.zeros(shape=shape, dtype='complex128')
    hopping_block = numpy.zeros(shape=shape, dtype='complex128')
    for i in range(block_size - 1):
        onsite_block[i, i + 1] = coupling
    for i in range(block_size):
        hopping_block[i, i] = coupling

    norbitals = nblocks * block_size
    mat = numpy.zeros(shape=(norbitals, norbitals), dtype='complex128')
    # Onsite blocks (upper hamiltonian).
    for i in range(nblocks):
        mat[
            i * block_size: (i + 1) * block_size,
            i * block_size: (i + 1) * block_size] = onsite_block

    # Hopping blocks until second contact.
    for i in range(nblocks - n_contact_blocks - 1):
        mat[
            i * block_size: (i + 1) * block_size,
            (i + 1) * block_size: (i + 2) * block_size] = hopping_block
    # Second contact.
    left_contact_index = (nblocks - n_contact_blocks) * block_size
    mat[
        left_contact_index: left_contact_index + block_size,
        0: block_size] = hopping_block

    for i in range(nblocks - n_contact_blocks + 1, nblocks):
        mat[
            (i - 1) * block_size: i * block_size,
            i * block_size: (i + 1) * block_size] = hopping_block

    mat_csr = sparse.csr_matrix(mat)
    mat_csr = mat_csr + mat_csr.conjugate(copy=True).transpose()
    mat_csr.sort_indices()

    return mat_csr
