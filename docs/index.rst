.. pynegf documentation master file, created by
   sphinx-quickstart on Sun Dec  8 16:07:40 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pynegf's documentation!
==================================

Pynegf is a Python wrapper around `libnegf <https://github.com/libnegf/libnegf>`_,
a Fortran library for the simulation of nanoelectronic devices based on the
Non Equilibrium Green's Function (NEGF) method.

`libnegf <https://github.com/libnegf/libnegf>`_ is used in
`DFTB+ <https://dftbplus-develguide.readthedocs.io/en/latest/>`_,
the purpose of this wrapper is to allow using the NEGF algorithm with custom
Hamiltonian or with external electronic structure codes.

Pynegf allows to use most of libnegf features. Currently this include:

- Calculation of transmission, coherent current and density of states via
  equilibium Green's function.

- Calculation of density matrix via complex contour integration using
  non-equilibrium green's function.

- Linear scaling for quasi-1d systems via iterative algorithm.

- Inclusion of electron-phonon interaction in the limit of dephasing
  approximation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   contents/gettingstarted.rst

   modules.rst

   notebooks/linearchain.ipynb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
