# Example test file.

from numpy.testing import assert_equal
import skprocrustes

def test_version_good():
    assert_equal(skprocrustes.__version__, "0.1")
