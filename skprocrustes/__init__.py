from .skprocrustes import ProcrustesProblem
from .skprocrustes import OptimizeResult
from .skprocrustes import ProcrustesSolver
from .skprocrustes import SPGSolver
from .skprocrustes import GKBSolver
from .skprocrustes import EBSolver
from .skprocrustes import GPISolver
from .skprocrustes import GBBSolver
from .skprocrustes import gkb_setup
from .skprocrustes import spectral_solver
from .skprocrustes import eb_solver
from .skprocrustes import gpi_solver
from .skprocrustes import gbb_solver
from .skprocrustes import blockbidiag
from .skprocrustes import bidiaggs
from .skprocrustes import compare_solvers
from .skprocrustes import compute_residual
from numpy.testing import Tester

__all__ = ['ProcrustesProblem', 'OptimizeResult', 'ProcrustesSolver',
           'SPGSolver', 'GKBSolver', 'EBSolver', 'GPISolver', 'GBBSolver',
           'gkb_setup', 'spectral_solver', 'eb_solver', 'gpi_solver',
           'gbb_solver', 'blockbidiag', 'bidiaggs', 'compare_solvers',
           'compute_residual']

__version__ = "0.1"
# If you want to use Numpy's testing framerwork, use the following.
# Tests go under directory tests/, benchmarks under directory benchmarks/

test = Tester().test
# bench = Tester().bench
