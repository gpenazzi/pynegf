Getting started
==================

pynegf is a Python wrapper around `libnegf <https://github.com/libnegf/libnegf>`_,
a Fortran library for the simulation of nanoelectronic devices based on the
Non Equilibrium Green's Function (NEGF) method.

`libnegf <https://github.com/libnegf/libnegf>`_ is used in `DFTB+ <https://dftbplus-develguide.readthedocs.io/en/latest/>`,
the purpose of this wrapper is to allow using the NEGF algorithm with custom
Hamiltonian or with external electronic structure codes.

To get started you must have installed libnegf on your system. This dependency
is not provided by the package yet.

Pynegf allows to use most of libnegf features. Currently this include:

- Calculation of transmission, coherent current and density of states via
  equilibium Green's function.

- Calculation of density matrix via complex contour integration using
  non-equilibrium green's function.

- Linear scaling for quasi-1d systems via iterative algorithm.

- Inclusion of electron-phonon interaction in the limit of dephasing
  approximation.

The python wrapper works similarly to the Fortran library: the Hamiltonian and
Overlap are passed as input parameters using setters.
Most of input parameters are contained in a data structure, first populated with
default values. The actual calculation is then triggerred with `solve` methods
and afterwards results can be retrieved via specialized functions.

The following example shows a calculation of transmission and DOS
for a simple linear chain.

.. literalinclude:: example1.py
   :language: python
