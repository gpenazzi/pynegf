import numpy
import pytest

import pynegf
from pynegf_test import utils

# Skip if libnegf is not available.
if pynegf.cdll_libnegf() is None:
    pytest.skip(
        "libnegf backengine not available on the system",
        allow_module_level=True)


def test_construction():
    """
    Test that we can construct a NEGF instance, without performing
    any operation.
    """
    foo = pynegf.PyNegf()
    assert(foo is not None)


def _check_transmission_values(
        transmission,
        energies,
        bandwidth=(-2.0, 2.0),
        value=1.0,
        energy_tolerance=0.01):
    """
    Check that we have defined transmission within the given bandwidth.
    """
    in_band_transmission = transmission[0, :][
        numpy.where(
            numpy.logical_and(
                numpy.real(energies) > bandwidth[0],
                numpy.real(energies) < bandwidth[1]))]
    out_band_transmission = transmission[0, :][
        numpy.where(
            numpy.logical_or(
                numpy.real(energies) < bandwidth[0],
                numpy.real(energies) > bandwidth[1]))]
    assert in_band_transmission == pytest.approx(value, abs=1e-4)
    assert out_band_transmission == pytest.approx(0.0, abs=1e-4)


def test_transmission_linear_chain():
    """ Test that we can calculate the transmission for an ideal linear chain. """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_linear_chain(
        nsites=100, contact_size=20, coupling=1.0)

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
    negf.params.g_spin = 1
    negf.params.emin = -3.0
    negf.params.emax = 3.0
    negf.params.estep = 0.01
    negf.params.mu[0] = -0.5
    negf.params.mu[1] = 0.5
    negf.set_params()
    # negf.print_tnegf()

    # Set also some local DOS intervals.
    negf.set_ldos_intervals(numpy.array([0, 30, 0]), numpy.array([59, 59, 29]))

    negf.solve_landauer()

    # Get transmission, dos and energies as numpy object
    energies = negf.energies()
    transmission = negf.transmission()
    ldos = negf.ldos()
    # The system is homogeneous, therefore the first LDOS should be equal to
    # the sum of the second and third.
    assert ldos[0, :] == pytest.approx(ldos[1, :] + ldos[2, :])
    # For sanity check also that the dos is positive and non all zero.
    assert numpy.sum(ldos) > 1.0

    # The current in ballistic current units should be equal the
    # spin degeneracy we set, i.e. 1
    currents = negf.currents()
    currents[0] == pytest.approx(1.0)

    # Check that the transmission has the right integer values.
    _check_transmission_values(transmission, energies)


def test_transmission_automatic_partition():
    """ Test that we can calculate the transmission for an ideal linear chain. """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_linear_chain(
        nsites=100, contact_size=20, coupling=1.0)

    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(100)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array([79, 99]),
        numpy.array([59, 79]))

    # Initialize parameters relevant for the transmission.
    negf.params.g_spin = 1
    negf.params.emin = -3.0
    negf.params.emax = 3.0
    negf.params.estep = 0.01
    negf.params.mu[0] = 0.1
    negf.set_params()
    # negf.print_tnegf()

    # Set also some local DOS intervals.
    negf.set_ldos_intervals(numpy.array([0, 30, 0]), numpy.array([59, 59, 29]))

    negf.solve_landauer()

    # Get transmission, dos and energies as numpy object
    energies = negf.energies()
    transmission = negf.transmission()

    ldos = negf.ldos()
    # The system is homogeneous, therefore the first LDOS should be equal to
    # the sum of the second and third.
    assert ldos[0, :] == pytest.approx(ldos[1, :] + ldos[2, :])
    # For sanity check also that the dos is positive and non all zero.
    assert numpy.sum(ldos) > 1.0

    # The current in ballistic current units should be equal the
    # spin degeneracy we set, i.e. 1
    currents = negf.currents()
    currents[0] == pytest.approx(1.0)

    # Check that the transmission has the right integer values.
    _check_transmission_values(transmission, energies)


def test_density_linear_chain_eq():
    """
    Test the density matrix calculation at equilibrium for a linear
    chain in full-band.
    """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_linear_chain(
        nsites=100, contact_size=20, coupling=1.0)

    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(100)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array([79, 99]),
        numpy.array([59, 79]))

    # Set parameters relevant for the density matrix calculation.
    # We fully occupy all band and use mostly default values for
    # the integration contour.
    negf.params.ec = -2.5
    negf.params.mu[0] = 0.0
    negf.params.mu[1] = 0.0
    negf.params.kbt_dm = (.001, .001)
    negf.params.g_spin = 2.0
    # Not correctly initialized, setting explicitely.
    negf.params.np_real = tuple([0] * 11)
    negf.params.verbose = 0

    negf.set_params()

    # Calculate the density matrix.
    negf.solve_density()
    density_matrix = negf.density_matrix()
    # We should have 1 particles (2 degeneracy, half band occupied) per site.
    diagonal = density_matrix.diagonal()
    assert diagonal[:60] == pytest.approx(1.0)

    # The contact density matrix is ignored, therefore it should be zero.
    assert diagonal[60:] == pytest.approx(0.0)


def test_density_linear_chain_neq_bias():
    """
    Test the density matrix calculation at non-equilibrium for a linear
    chain in full-band.
    """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_linear_chain(
        nsites=100, contact_size=20, coupling=1.0)

    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(100)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array([79, 99]),
        numpy.array([59, 79]))

    # Set parameters relevant for the density matrix calculation.
    # We fully occupy all band and use mostly default values for
    # the integration contour.
    negf.params.ec = -2.5
    negf.params.mu[0] = 0.1
    negf.params.mu[1] = -0.1
    negf.params.kbt_dm = (.001, .001)
    negf.params.g_spin = 2.0
    # Not correctly initialized, setting explicitely.
    negf.params.np_real = tuple([0] * 11)
    negf.params.verbose = 0

    negf.set_params()

    # Calculate the density matrix.
    negf.solve_density()
    density_matrix = negf.density_matrix()
    diagonal = density_matrix.diagonal()

    # The system is ballistic, therefore we should have identical
    # occupation all over the chain.
    assert diagonal[:60] == pytest.approx(diagonal[0])

    # The occupation should be slighlty above 1.0 (equilibrium case).
    assert 0.01 < diagonal[0] - 1.0 < 0.05

    # We should have 2 particles (due to degeneracy) per site.
    diagonal = density_matrix.diagonal()
    print(diagonal)

    # The contact density matrix is ignored, therefore it should be zero.
    assert diagonal[60:] == pytest.approx(0.0)
