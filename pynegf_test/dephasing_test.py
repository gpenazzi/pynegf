import numpy
import pytest

import pynegf
from pynegf_test import utils

# Skip if libnegf is not available.
if pynegf.cdll_libnegf() is None:
    pytest.skip(
        "libnegf backengine not available on the system",
        allow_module_level=True)


def _transmission_linear_chain_dephasing(coupling=None):
    """
    Utility to calculate the transmission in presence of diagonal
    dephasing for a nearest neighbor linear chain.
    """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_linear_chain(
        nsites=100, contact_size=10, coupling=1.0)

    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(100)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array([89, 99]),
        numpy.array([79, 89]))

    # Initialize parameters relevant for the transmission.
    negf.params.g_spin = 1
    negf.params.emin = -2.5
    negf.params.emax = 2.5
    negf.params.estep = 0.025
    negf.params.mu[0] = 2.1
    negf.params.mu[1] = -2.1
    negf.verbosity = 0
    negf.set_params()
    if coupling is not None:
        negf.set_diagonal_elph_dephasing(numpy.array([coupling]*80))

    negf.solve_landauer()

    # Get transmission, dos and energies as numpy object
    energies = negf.energies()
    if coupling is None:
        transmission = negf.transmission()
    else:
        transmission = negf.energy_current()

    return energies, transmission


def test_transmission_dephasing_linear_chain():
    """
    Test that we can calculate the transmission with dephasing for an
    ideal linear chain.
    """
    energies, ballistic_transmission = _transmission_linear_chain_dephasing()

    dephasing_transmissions = []
    for coupling in [0.0, 0.01, 0.05]:
        energies, transmission = _transmission_linear_chain_dephasing(
            coupling=coupling)
        dephasing_transmissions.append(transmission)

    # The ballistic transmission should be equal to the dephasing
    # case with zero coupling.
    assert numpy.linalg.norm(
        ballistic_transmission - dephasing_transmissions[0]
        ) == pytest.approx(0.)

    # Increasing the coupling, the transmission should go lower.
    tol = 0.001
    assert (dephasing_transmissions[1] < dephasing_transmissions[0] + tol).all()
    assert (dephasing_transmissions[2] < dephasing_transmissions[1] + tol).all()

    # A quantitative check on the mid-point.
    mid_point = energies.size // 2
    assert dephasing_transmissions[0][0, mid_point] == pytest.approx(1.0)
    assert dephasing_transmissions[1][0, mid_point] == pytest.approx(0.999, abs=1e-3)
    assert dephasing_transmissions[2][0, mid_point] == pytest.approx(0.95, abs=1e-2)


def _density_linear_chain_dephasing(coupling=None, orthogonal=True):
    """
    Utility to calculate the density matrix in presence of diagonal
    dephasing for a nearest neighbor linear chain.
    """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_linear_chain(
        nsites=50, contact_size=10, coupling=1.0)
    if orthogonal:
        negf.set_hamiltonian(mat_csr)
        # Set an identity overlap matrix.
        negf.set_identity_overlap(50)
    else:
        negf.set_hamiltonian(mat_csr)
        mat_csr = utils.orthogonal_linear_chain(
            nsites=50, contact_size=10, coupling=0.1, onsite=1.0)
        # This is to make sure that S is positive definite.
        numpy.linalg.cholesky(mat_csr.todense())
        # Set an identity overlap matrix.
        negf.set_overlap(mat_csr)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array([39, 49]),
        numpy.array([29, 39]))

    # Initialize parameters relevant for the density matrix calculation.
    negf.params.ec = -3.5
    negf.params.mu[0] = -0.1
    negf.params.mu[1] = 0.1
    negf.params.kbt_dm[0] = 0.001
    negf.params.kbt_dm[1] = 0.001
    negf.params.np_real[0] = 50
    negf.params.verbose = 100
    negf.set_params()
    if coupling is not None:
        negf.set_diagonal_elph_dephasing(numpy.array([coupling]*30))

    negf.solve_density()

    # Get the density matrix.
    density_matrix = negf.density_matrix()

    return density_matrix


def test_density_matrix_dephasing_linear_chain():
    """
    Test that we can calculate the density matrix with dephasing for an
    ideal linear chain.
    """
    ballistic_density_matrix = _density_linear_chain_dephasing()
    ballistic_density_matrix = ballistic_density_matrix.todense()

    dephasing_density_matrix = []
    for coupling in [0.0, 0.05, 0.5]:
        density_matrix = _density_linear_chain_dephasing(
            coupling=coupling, orthogonal=True)
        dephasing_density_matrix.append(density_matrix.todense())

    # The ballistic density matrix should be equal to the dephasing
    # case with zero coupling.
    assert numpy.linalg.norm(
        ballistic_density_matrix - dephasing_density_matrix[0]
        ) == pytest.approx(0.)

    # In presens of dephasing the occupation should be a ramp,
    # decreasing from the left to the right contact.
    for dm in dephasing_density_matrix[1:]:
        for i in range(1, 29):
            assert numpy.diagonal(dm)[i - 1] > numpy.diagonal(dm)[i]

    # The difference in density between first and last device site
    # should increase with increasing dephasing.
    dm1 = numpy.real(numpy.diagonal(dephasing_density_matrix[1]))
    dm2 = numpy.real(numpy.diagonal(dephasing_density_matrix[2]))
    assert ((dm1[0] - dm1[29]) < (dm2[0] - dm2[29]))

    # Mid-value check. The density ramp should cross the ballistic
    # one at half chain length.
    assert (all(dm1[:15] > 1.0) and all(dm1[15:] < 1.0))
    assert (all(dm2[:15] > 1.0) and all(dm2[15:] < 1.0))

    # Approximate delta determined by inspection, for regression.
    assert dm1[0] - dm1[29] == pytest.approx(0.00115, rel=0.01)
    assert dm2[0] - dm2[29] == pytest.approx(0.0399, rel=0.01)


@pytest.mark.skip(reason="Not clear if the backengine is correct physically")
def test_density_matrix_dephasing_linear_chain_overlap():
    """
    Test that we can calculate the density matrix with dephasing for an
    ideal linear chain with overlap.
    """
    ballistic_density_matrix = _density_linear_chain_dephasing(
        orthogonal=False)
    ballistic_density_matrix = ballistic_density_matrix.todense()

    dephasing_density_matrix = []
    for coupling in [0.0, 0.1, 0.5]:
        density_matrix = _density_linear_chain_dephasing(
            coupling=coupling, orthogonal=False)
        dephasing_density_matrix.append(density_matrix.todense())

    # The ballistic density matrix should be equal to the dephasing
    # case with zero coupling.
    assert numpy.linalg.norm(
        ballistic_density_matrix - dephasing_density_matrix[0]
        ) == pytest.approx(0.)

    # In presens of dephasing the occupation should be a ramp,
    # decreasing from the left to the right contact.
    for dm in dephasing_density_matrix[1:]:
        for i in range(1, 29):
            assert numpy.diagonal(dm)[i - 1] > numpy.diagonal(dm)[i]

    # The difference in density between first and last device site
    # should increase with increasing dephasing.
    dm1 = numpy.real(numpy.diagonal(dephasing_density_matrix[1]))
    dm2 = numpy.real(numpy.diagonal(dephasing_density_matrix[2]))
    assert ((dm1[0] - dm1[29]) < (dm2[0] - dm2[29]))

    # Mid-value check. The density ramp should cross the ballistic
    # one at half chain length.
    assert (all(dm1[:15] > 1.0) and all(dm1[15:] < 1.0))
    assert (all(dm2[:15] > 1.0) and all(dm2[15:] < 1.0))

    # Approximate delta determined by inspection, for regression.
    # TODO: values to be determined
    # assert dm1[0] - dm1[29] == pytest.approx(0.00115, rel=0.01)
    # assert dm2[0] - dm2[29] == pytest.approx(0.0399, rel=0.01)


def test_current_conservation_dephasing(coupling=None):
    """ Test that we have current conservation at the electrodes """
    currents = []
    for orthogonal in [True, False]:
        for (ni, nf) in [(1, 2), (2, 1)]:
            negf = pynegf.PyNegf()
            # Build the sparse hamiltonian for the nearest-neighbor linear chain.
            mat_csr = utils.orthogonal_linear_chain(
                nsites=50, contact_size=10, coupling=1.0)
            if orthogonal:
                negf.set_hamiltonian(mat_csr)
                # Set an identity overlap matrix.
                negf.set_identity_overlap(50)
            else:
                negf.set_hamiltonian(mat_csr)
                mat_csr = utils.orthogonal_linear_chain(
                    nsites=50, contact_size=10, coupling=0.1, onsite=1.0)
                # This is to make sure that S is positive definite.
                numpy.linalg.cholesky(mat_csr.todense())
                # Set an identity overlap matrix.
                negf.set_overlap(mat_csr)

            # Initialize the system structure.
            negf.init_structure(
                2,
                numpy.array([39, 49]),
                numpy.array([29, 39]))

            # Initialize parameters relevant for the transmission.
            negf.params.ec = -5.0
            negf.params.emin = -0.2
            negf.params.emax = 0.2
            negf.params.estep = 0.005
            negf.params.mu[0] = 0.1
            negf.params.mu[1] = -0.1
            # IMPORTANT: ni and nf must be both defined.
            negf.params.ni[0] = ni
            negf.params.ni[1] = nf
            negf.params.nf[0] = nf
            negf.params.nf[1] = ni
            negf.params.kbt_t[1] = 0.01
            negf.params.kbt_t[0] = 0.01
            negf.params.np_real[0] = 50
            negf.verbosity = 0
            negf.set_params()
            coupling = 0.1
            negf.set_diagonal_elph_dephasing(numpy.array([coupling]*30))

            negf.solve_landauer()
            tmp_currents = negf.currents()
            assert tmp_currents[0] == pytest.approx( -tmp_currents[1], 1e-9)
            currents.append(tmp_currents[0])

        assert currents[0] == pytest.approx(-currents[1], 1e-9)
