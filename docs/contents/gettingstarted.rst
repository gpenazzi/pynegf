Getting started
==================

Pynegf requires a working installation of
`libnegf <https://github.com/libnegf/libnegf>`_. Pynegf ships
his own version of libnegf as submodule, as it might occasionally
point at experimental forks ahead of the official release.

To clone the submodule run in the pynegf root directory:

::
    $ git submodule init
    $ git submodule clone


The compilation of the distributed submodule is driven by
`scikit-build <https://github.com/scikit-build/scikit-build>`_.
Installing pynegf should be as simple as running

::

    $ pip install .

the package can also be installed with `pip install -e` (editable mode)
for development purpose. In this case, make sure to add the location
of the generated `libnegf.so` to `LD_LIBRARY_PATH`.
Pynegf requires also an installed version of blas and lapack, which are
loaded at runtime.

It is also possible to install libnegf in the system, and pynegf will
try to locate it automatically.
You can force a given libnegf path doing the following:

::

    $ import pynegf
    $ pynegf.settings['negf']=r'/home/user/whateverpath/libnegf.so'
    $ pynegf.dependencies = load_dependencies()


The python wrapper works similarly to the Fortran library: the Hamiltonian and
Overlap are passed as input parameters using setters.
Most of input parameters are contained in a data structure, first populated
with default values. The actual calculation is then triggerred with `solve`
methods and afterwards results can be retrieved via specialized functions.

The following example shows a calculation of transmission and DOS
for a simple linear chain.

.. literalinclude:: example1.py
   :language: python
