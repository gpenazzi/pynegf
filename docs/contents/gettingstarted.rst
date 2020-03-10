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
for development purpose. Depending on your system settings, you might need
to add the generated `libnegf.so` to `LD_LIBRARY_PATH`.
The package can also be installed using `setup.py`, which might grant
more flexibility in passing flags to cmake. For example, to
compile libnegf with MPI support you can use:
::

    $ python setup.py develop -- -DWITH_MPI=ON

For more information on how to pass additional arguments refer to the
`scikit-build <https://github.com/scikit-build/scikit-build>`_ documentation.

Pynegf requires also an installed version of blas and lapack, which are
loaded at runtime.

It is also possible to install libnegf and skip the libnegf build step
in installation.
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
