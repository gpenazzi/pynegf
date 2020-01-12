Getting started
==================

To get started you must have installed `libnegf <https://github.com/libnegf/libnegf>`_
on your system. libnegf is also shipped as submodule of pynegf.
It is suggested to build from the submodule to ensure consistency between the
wrapper and the library.

To clone the submodule run in the pynegf root directory:

::
    $ git submodule init
    $ git submodule clone


To compile libnegf refer to the library documentation. The simplest way is
to use cmake:

::

    $ cd libnegf && mkdir _build && cd _build
    $ cmake -DBUILD_SHARED_LIBS ..
    $ make
    $ make install

If you install in a directory with root privileges, you will need to run
`make install` as admin or with sudo.
Pynegf will then try to determine libnegf location automatically.

The python wrapper works similarly to the Fortran library: the Hamiltonian and
Overlap are passed as input parameters using setters.
Most of input parameters are contained in a data structure, first populated
with default values. The actual calculation is then triggerred with `solve`
methods and afterwards results can be retrieved via specialized functions.

The following example shows a calculation of transmission and DOS
for a simple linear chain.

.. literalinclude:: example1.py
   :language: python
