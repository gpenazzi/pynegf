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
        value=1.0):
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


def _check_ldos_shape(
        ldos,
        energies,
        bandwidth=(-2.0, 2.0),
        peak_position_tolerance=0.05,
        energy_tolerance=0.05):
    """
    Check that the dos is non-zero in the band and that the highest
    values is where we expect the Van-Hove singularities.
    """
    for ldos_row in ldos:
        in_band_ldos = ldos_row[
            numpy.where(
                numpy.logical_and(
                    numpy.real(energies) > bandwidth[0] + energy_tolerance,
                    numpy.real(energies) < bandwidth[1] - energy_tolerance))]
        out_band_ldos = ldos_row[
            numpy.where(
                numpy.logical_or(
                    numpy.real(energies) < bandwidth[0] - energy_tolerance,
                    numpy.real(energies) > bandwidth[1] + energy_tolerance))]
        assert out_band_ldos == pytest.approx(0.0, abs=1e-2)
        peak_position = numpy.argmax(ldos_row)
        assert (
            energies[peak_position] == pytest.approx(
                bandwidth[0], abs=peak_position_tolerance)
            or energies[peak_position] == pytest.approx(
                bandwidth[1], abs=peak_position_tolerance))
        # The lowest in-band value should be at the middle of the band.
        minimum_position = numpy.argmin(
            in_band_ldos)
        in_band_energies = energies[
            numpy.where(
                numpy.logical_and(
                    numpy.real(energies) > bandwidth[0] + energy_tolerance,
                    numpy.real(energies) < bandwidth[1] - energy_tolerance))]
        assert (
            in_band_energies[minimum_position] == pytest.approx(0.0))


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
    # Check ldos shape.
    _check_ldos_shape(ldos, energies)

    # The current in ballistic current units should be equal the
    # spin degeneracy we set, i.e. 1
    currents = negf.currents()
    currents[0] == pytest.approx(1.0)

    # Check that the transmission has the right integer values.
    _check_transmission_values(transmission, energies)


def test_transmission_automatic_partition():
    """
    Test that we can calculate the transmission for an ideal linear chain.
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
    # Check ldos shape.
    _check_ldos_shape(ldos, energies)

    # The current in ballistic current units should be equal the
    # spin degeneracy we set, i.e. 1
    currents = negf.currents()
    currents[0] == pytest.approx(1.0)

    # Check that the transmission has the right integer values.
    _check_transmission_values(transmission, energies)


def test_transmission_automatic_partition_2d():
    """
    Test that we can calculate the transmission for a 2d hamiltonian.
    """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_square_2d_lattice(
        nblocks=20, block_size=5, n_contact_blocks=2, coupling=1.0)

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
    negf.params.emin = -5.0
    negf.params.emax = 5.0
    negf.params.estep = 0.01
    negf.params.mu[0] = 0.05
    negf.params.mu[0] = -0.05
    negf.set_params()
    # negf.print_tnegf()

    # Set also some local DOS intervals.
    negf.set_ldos_intervals(numpy.array([0, 30, 0]), numpy.array([59, 59, 29]))

    negf.solve_landauer()

    # Get transmission, dos and energies as numpy object
    transmission = negf.transmission()
    ldos = negf.ldos()

    # The system is homogeneous, therefore the first LDOS should be equal to
    # the sum of the second and third.
    assert ldos[0, :] == pytest.approx(ldos[1, :] + ldos[2, :])

    # The current in ballistic current units should be equal the
    # spin degeneracy we set times 5 (number of bands) times delta_mu, i.e. 0.5
    currents = negf.currents()
    currents[0] == pytest.approx(0.5)

    # Check that the transmission peaks at 5.0.
    assert numpy.max(transmission[0, :]) == pytest.approx(5.0)


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


def test_density_linear_chain_eq_2d():
    """
    Test the density matrix calculation at equilibrium for a 2d lattice
    chain in full-band.
    """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_square_2d_lattice(
        nblocks=20, block_size=5, n_contact_blocks=2, coupling=1.0)

    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(100)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array([89, 99]),
        numpy.array([79, 89]))

    # Set parameters relevant for the density matrix calculation.
    # We fully occupy all band and use mostly default values for
    # the integration contour.
    negf.params.ec = -5.0
    negf.params.mu[0] = 0.0
    negf.params.mu[1] = 0.0
    negf.params.kbt_dm = (.001, .001)
    negf.params.g_spin = 2.0
    negf.params.verbose = 0

    negf.set_params()

    # Calculate the density matrix.
    negf.solve_density()
    density_matrix = negf.density_matrix()
    # We should have 1 particles (2 degeneracy, half band occupied) per site.
    diagonal = density_matrix.diagonal()
    assert diagonal[:80] == pytest.approx(1.0)

    # The contact density matrix is ignored, therefore it should be zero.
    assert diagonal[80:] == pytest.approx(0.0)


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
    negf.params.np_real[0] = 50
    negf.params.verbose = 0

    negf.set_params()

    # Calculate the density matrix.
    negf.solve_density()
    density_matrix = negf.density_matrix()
    diagonal = density_matrix.diagonal()

    # The system is ballistic, therefore we should have identical
    # occupation all over the chain.
    assert diagonal[:60] == pytest.approx(diagonal[0], abs=1e-4)

    # The occupation should 1.
    assert diagonal[0] == pytest.approx(1.0, abs=1e-4)

    # We should have 2 particles (due to degeneracy) per site.
    diagonal = density_matrix.diagonal()

    # The contact density matrix is ignored, therefore it should be zero.
    assert diagonal[60:] == pytest.approx(0.0)


def test_density_linear_chain_neq_2d():
    """
    Test the density matrix calculation at non-equilibrium for a linear
    chain in full-band.
    """
    negf = pynegf.PyNegf()
    # Build the sparse hamiltonian for the nearest-neighbor linear chain.
    mat_csr = utils.orthogonal_square_2d_lattice(
        nblocks=20, block_size=5, n_contact_blocks=2, coupling=1.0)

    negf.set_hamiltonian(mat_csr)

    # Set an identity overlap matrix.
    negf.set_identity_overlap(100)

    # Initialize the system structure.
    negf.init_structure(
        2,
        numpy.array([89, 99]),
        numpy.array([79, 89]))

    # Set parameters relevant for the density matrix calculation.
    # We fully occupy all band and use mostly default values for
    # the integration contour.
    negf.params.ec = -5.0
    negf.params.mu[0] = 0.1
    negf.params.mu[1] = -0.1
    negf.params.kbt_dm = (.001, .001)
    negf.params.g_spin = 2.0
    # Not correctly initialized, setting explicitely.
    negf.params.np_real = tuple([50] * 11)
    negf.params.verbose = 0

    negf.set_params()

    # Calculate the density matrix.
    negf.solve_density()
    density_matrix = negf.density_matrix()
    diagonal = density_matrix.diagonal()

    # The system is ballistic, therefore we should have identical
    # occupation all over the chain when checking equivalent sites.
    for i in range(5):
        assert diagonal[i:80:5] == pytest.approx(diagonal[i], abs=1e-4)

    # The occupation should 1.
    assert diagonal[0] == pytest.approx(1.0, abs=1e-3)

    # We should have 2 particles (due to degeneracy) per site.
    diagonal = density_matrix.diagonal()

    # The contact density matrix is ignored, therefore it should be zero.
    assert diagonal[80:] == pytest.approx(0.0)
