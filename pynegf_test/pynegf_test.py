import pynegf

import pytest

# Skip if libnegf is not available.
try:
    pynegf.load_libnegf()
except OSError:
    _ = pytest.skip("libnegf backengine not available on the system", allow_module_level=True)


def test_construction():
    """ Test that we can construct a NEGF instance """
    foo = pynegf.PyNegf()
    assert(foo is not None)
    