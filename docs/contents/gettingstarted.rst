Getting started
==================

To get started you must have installed libnegf on your system. This dependency
is not provided by the package yet. In order to be imported, libnegf
has to be compiled as dynamic library. This is as simple as doing:

::

    $ git clone https://github.com/libnegf/libnegf.git
    $ cd libnegf && mkdir _build && cd _build
    $ cmake -DBUILD_SHARED_LIBS ..
    $ make
    $ make install

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
