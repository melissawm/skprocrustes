# Solvers for the Weighted Orthogonal Procrustes Problem (WOPP)
#   min norm(AXC-B, 'fro')**2
#   s.t. X^TX=I
# where A(m,n), B(m,q), C(p,q) and X(n,p). (usually n >> p, which
# means we can solve unbalanced problems.)
#
# Usage:
#     >>> import skprocrustes as skp
#     >>> problem = skp.ProcrustesProblem((m,n,p,q),
#                                         problemnumber=[1,2,3], # or
#                                         matrices=[ndarrays]))
#     >>> solver = skp.SPGSolver(**kwargs)
#     >>> result = solver.solve(problem)
#
# Solvers currently implemented:
#
#   > SPGSolver: spectral projected gradient method - no blocks
#                uses full SVD at each outer iteration
#   > GKBSolver: spectral projected gradient method - block GKB
#                uses block Lanczos bidiagonalization to gradually
#                increase the problem size and hopefully solve the
#                Procrustes problem before full matrix is used.
#   > EBSolver : expansion-balance method
#   > GPISolver: Generalized Power Iteration method for WOPP
#
# All solvers listed above extend ProcrustesSolver, which is the base
# solver class. In adition, GKBSolver also extends SPGSolver.
#
# See the documentation for more details.

import numpy as np
# The subpackages of scipy do not get imported by import scipy by design. It 
# is intended that you import them separately. Never use a plain import scipy.
from scipy import linalg as sp
from scipy.sparse import linalg as spl
import matplotlib.pyplot as plt
import time

# standard status messages of optimizers (based on scipy.optimize)
_status_message = {'success'     : 'Optimization terminated successfully.',
                   'infeasible'  : 'Infeasible point found.',
                   'smallpred'   : 'Small PRED',
                   'negativepred': 'Negative PRED',
                   'maxiter'     : 'Maximum number of iterations has been '
                   'exceeded.'}

class ProcrustesProblem:

    """
    The problem we want to solve.

    Usage example (default problem):

       >>> import skprocrustes as skp
       >>> problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)

    Usage example (user defined problem):

       >>> import skprocrustes as skp
       >>> import numpy as np
       >>> A = ... # given by the user
       >>> B = ... # given by the user
       >>> C = ... # given by the user
       >>> X = ... # given by the user (optional)
       >>> problem = skp.ProcrustesProblem((m,n,p,q), matrices=(A,B,C,X))

    Input Parameters:
    
       ``sizes``: tuple
          ``(m,n,p,q)``, where :math:`A_{m\\times n}, B_{m \\times q}, 
          C_{p\\times q}` and :math:`X_{n\\times p}`.
       
       *(optional)* ``problemnumber``: int
          Can be ``1``, ``2`` or ``3``, and selects one of the predefined
          problems as described in reference :ref:`[4] <bibliography>`.
          (for more details, see the documentation for ``_setproblem``)

       *(optional)* ``matrices``: list of ndarrays
          If present, must contain a list of three or four matrices 
          corresponding to :math:`A`, :math:`B`, :math:`C`, and optionally
          :math:`X` (known solution) with adequate sizes.

    .. note::
       Currently, m must be equal do n, and p must be equal do q. This 
       is the case for all three solvers. (However, n can be greater than
       p)

    .. note::
       If ``matrices`` is not given by the user, ``problemnumber``
       (1, 2 or 3) must be selected so that one of the default problems 
       is built. 

    Attributes:

    The problem matrices (generated or given) are accessible via

       >>> problem.A
       >>> problem.B
       >>> problem.C
       >>> problem.Xsol
    """
    
    def __init__(self, sizes, problemnumber=None, matrices=[]):
        if sizes[0] != sizes[1] or sizes[2] != sizes[3]:
            raise Exception("Currently only square problems are supported.")
        elif sizes[0] % sizes[2] != 0:
            raise Exception("Dimensions are incompatible: m must be "
                            "divisible by p.")
        else:
            self.sizes = sizes
        self.problemnumber = problemnumber
        self._setproblem(matrices, problemnumber)
        
        # stats will be filled when the optimization is finished and
        # will be returned in the OptimizeResult instance.
        self.stats = dict([])
        # { "nbiter": 0,
        #   "svd": 0,
        #   "fev": 0,
        #   "gradient": 0,
        #   "blocksteps": 0,
        #   "full_results": [None]}
        
    def _setproblem(self, matrices, problemnumber):

        """
        Method to effectively build A, B, and C if they are not already given.

        *This method should not be called directly; it is called by the 
        ProcrustesProblem constructor.*

        Available problems are all based on reference :ref:`[4] <bibliography>`:
        All problems have the form 

           .. math::

              A = U\Sigma V^T

        where :math:`\Sigma` varies between problems, and

           .. math::
        
              U = I_{m\\times m} - 2uu^T\\\\
              V = I_{n\\times n} - 2vv^T

        where :math:`u` and :math:`v` are randomly generated using 
        ``np.random.randn`` (normal distribution) and then normalized.
        
        :math:`C` can be built, but for our predefined problems it is always
        the identity matrix.

        ``problemnumber = 1``:
           Well conditioned problem: the singular values are randomly and 
           uniformly distributed in the interval [10,12].

        ``problemnumber = 2``:
           For this problem, the singular values are
        
              .. math::
         
                 \\sigma_i = 1 + \\frac{99(i-1)}{(m-1)} + 2r_i

           and :math:`r_i` are random numbers chosen from a uniform distribution
           on the interval [0,1].

        ``problemnumber = 3``:
           For this problem, the singular values are

              .. math::

                 \\sigma_i = \\left\\{ 
                 \\begin{align}
                 10 + r, & \\qquad 1\\leq i \\leq m_1\\\\
                 5 + r, & \\qquad m_1+1\\leq i \\leq m_2\\\\
                 2 + r, & \\qquad m_2+1\\leq i \\leq m_3\\\\
                 \\frac{r}{1000}, & \\qquad m_3+1\\leq i\\leq m
                 \\end{align}
                 \\right.

           Thus, :math:`A` has several small singular values and is 
           ill-conditioned.          

        .. note::

           ``problemnumber = 3`` can only be used if :math:`n = 50`, 
           :math:`n = 95`, :math:`n = 500` or :math:`n = 1000`.
        """
        # A(m,n), B(m,q), C(p,q) 
        m,n,p,q = self.sizes
        
        if not matrices:

            # Predefined examples for testing and benchmarking.
            # Reference:
            # [1] Z. Zhang, K. Du, Successive projection method for solving
            #     the unbalanced Procrustes problem, Sci. China Ser. A, 2006,
            #     49: 971–986.

            if problemnumber == 1:
                # Zhang & Du [1] - Example 1
                Sigmaorig = 10.0 + 2.0*np.random.rand(n)
                Sigma = np.zeros((m,n))
                Sigma[0:n, 0:n] = np.diag(Sigmaorig)

            elif problemnumber == 2:
                # Zhang & Du [1] - Example 3
                Sigmaorig = np.zeros(min(m,n))
                for i in range(0,min(m,n)):
                    Sigmaorig[i] = 1.0 + (99.0*float(i-1))/(float(n)-1.0) \
                                   + 2.0*np.random.rand(1)
                Sigma = np.zeros((m,n))
                Sigma[0:n, 0:n] = np.diag(Sigmaorig)

            elif problemnumber == 3:
                # Zhang & Du [1] - Example 4
                #
                # Only works if n = 50, n = 95, n = 500 or n = 1000
                #

                if n == 50:
                    n1, n2, n3, n4 = (15, 15, 12, 8)
                elif n == 95:
                    n1, n2, n3, n4 = (30, 30, 30, 5)
                elif n == 500:
                    n1, n2, n3, n4 = (160, 160, 160, 20)
                elif n == 1000:
                    n1, n2, n3, n4 = (300, 300, 240, 160)
                else:
                    raise Exception ("Error!!! Problem 6 requires n = 50,"
                                     "n = 95, n = 500 or n = 1000.")
                    
                vaux = np.zeros((n,1))
                vaux[0:n1] = 10.0 + np.random.rand(n1,1)
                vaux[n1:n1+n2] = 5.0 + np.random.rand(n2,1)
                vaux[n1+n2:n1+n2+n3] = 2.0 + np.random.rand(n3,1)
                vaux[n1+n2+n3:] = np.random.rand(n4,1)/1000.0

                Sigma = np.diagflat(vaux)
            else:
                raise Exception ("Problem matrices are empty and no "
                                 "problem number has been chosen.")

            # Building A
            uu = np.random.randn(m,1)
            U = np.eye(m,m) - 2.0*uu*uu.T/sp.norm(uu)**2
            
            vv = np.random.randn(n,1)
            V = np.eye(n,n) - 2.0*vv*vv.T/sp.norm(vv)**2

            A = np.dot(U,np.dot(Sigma,V.T))

            # Building C
            C = np.eye(p,q)
            
            ## Unused option:
            ## VcL = np.eye(p,p)
            ## VcR = np.eye(q,q)
            ## VcL = np.eye(p,p) - 2.0*uu[0:p]*uu[0:p].T/(sp.norm(uu[0:p]))**2
            ## VcR = np.eye(q,q) - 2.0*vv[0:q]*vv[0:q].T/(sp.norm(vv[0:q]))**2
            ## C = np.dot(VcL,np.dot(C,VcR.transpose()))
        
            # Known solution
            ## Unused option:
            ## real (wp) :: vsol(size(A,2))
            ## call random_number(vsol)
            ## vsol = vsol/dnrm2(n,vsol,1)
            ## Xsol1 = eye(n) - 2*vsol*vsol';

            Xsol = np.eye(n,p)
            Xsol = np.random.permutation(Xsol)
        
            # Building B
    
            B = np.dot(A, np.dot(Xsol, C))

            # Set problem up
            self.Xsol = Xsol
            self.A = A
            self.B = B
            self.C = C

        else:
            # User supplied problem.
            if len(matrices) < 3 or not (type(matrices) is tuple \
               or type(matrices) is list):
                raise Exception("matrices must be a list or tuple containing"
                                "matrices A, B and C (optionally, also a "
                                "known solution Xsol with compatible sizes:"
                                "A(m,n), B(m,q), C(p,q), Xsol(n,p)")
            else:
                if matrices[0].shape != (m,n):
                    raise Exception("A must be (m,n)")
                else:
                    self.A = matrices[0]

                if matrices[1].shape != (m,q):
                    raise Exception("B must be (m,q)")
                else:
                    self.B = matrices[1]

                if matrices[2].shape != (p,q):
                    raise Exception("C must be (p,q)")
                else:
                    self.C = matrices[2]
                    
            if len(matrices) == 4:
                if matrices[3].shape != (n,p):
                    raise Exception("Xsol must be (n,p)")
                else:
                    self.Xsol = matrices[3]

class OptimizeResult(dict):
    """ Represents the optimization result. 
    (*based on scipy.optimize.OptimizeResult*)

    This class is constructed as a dictionary of parameters defined by
    the creation of the instance. Thus, its attributes may vary.    

    Possible attributes:

    - ``success`` : ``bool``
       Whether or not the optimizer exited successfully.
    - ``status`` : ``int``
       Termination status of the optimizer. Its value depends on the
       underlying solver. Refer to `message` for details.
    - ``message`` : ``str``
       Description of the cause of the termination.
    - ``fun`` : ``float``
       Value of the objective function at the solution.
    - ``normgrad`` : ``float``
       Value of the norm of the gradient at the solution.
    - ``nbiter`` : ``int``
       Number of iterations performed by the optimizer.
    - ``nfev`` : ``int``/``float``
       Number of evaluations of the objective function (if called by 
       GKBSolver, nfev is a float representing the proportional number
       of calls to the objective function at each block step).
    - ``blocksteps`` : ``int``
       Number of blocksteps performed (if called by GKBSolver)
    - ``total_fun``: list
       List of objective function values for each iteration performed 
       (used to report and compare algorithms). Only if ``full_results``
       is True.
    - ``total_grad``: list
       List of gradient norm values for each iteration performed 
       (used to report and compare algorithms). Only if ``full_results``
       is True, and only for SPGSolver and GKBSolver.
    - ``total_crit``: list
       List of criticality measure values for each iteration performed 
       (used to report and compare algorithms). Only if ``full_results``
       is True, and only for EBSolver and GPISolver.

    Notes:
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

class ProcrustesSolver:
    """
    Abstract class to implement a solver for the ProcrustesProblem.

    All subclasses should implement the following methods:
    """
    def __init__(self, *args, **kwargs):
        self.solver = None
    
    def _setoptions(self, *args, **kwargs):
        """
        Choose which options are valid and applicable to this solver.
        """
        pass
            
    def solve(self, *args, **kwargs):
        """
        Call a solver function and set up the ``OptimizeResult`` instance with
        the result and statistics as convenient for this solver. Should be
        something like this:

        ::

           output = somesolver(problem, *args, **kwargs)
           result = OptimizeResult(output)
           return result

        """
        # output = self.solver(*args, **kwargs)
        output = dict([])
        result = OptimizeResult(output)
        return result

class SPGSolver(ProcrustesSolver):
    """
    Subclass containing the call to the ``spectral_setup()`` function 
    corresponding to the Spectral Projected Gradient solver described in 
    :ref:`[1] <bibliography>` and :ref:`[2] <bibliography>`.

    Usage example:

       >>> mysolver = skp.SPGSolver(verbose=3)
       >>> result = mysolver.solve(problem)

    Input:

    ``key = value``: keyword arguments available
       - ``full_results``: (*default*: ``False``)
          Return list of criticality values at each iteration (for later
          comparison between solvers)
       - ``strategy``: (*default*: ``"newfw"``)
          - ``"monotone"``: 
             monotone trust region
          - ``"bazfr"`` : 
             nonmonotone method according to :ref:`[1] <bibliography>`
          - ``"newfw"`` : 
             nonmonotone method according to :ref:`[2] <bibliography>`
       - ``gtol``: (*default*: ``1e-3``)
          tolerance for detecting convergence on the gradient
       - ``maxiter``: (*default*: ``2000``)
          maximum number of iterations allowed
       - ``verbose``: (*default*: ``1``)
          verbosity level. Current options:
          - ``0``: only convergence info
          - ``1``: only show time and final stats
          - ``2``: show outer iterations
          - ``3``: everything (except debug which is set separately)
       - ``changevar``: (*default*: ``False``)
          boolean option to allow for a change of variables before starting the 
          method. Currently disabled due to bad performance.

    Output:

    ``solver``: ``ProcrustesSolver`` instance
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self._setoptions(options=kwargs)
        self.solvername = "spg"
    
    def _setoptions(self, options):

        """
        Sets and validates options for the SPGSolver.

        *This method should not be called directly; it is called by the 
        SPGSolver constructor.*
        """

        # Options for the solver.
        # The user should not call this method explicitly, but pass the desired
        # options as arguments when instantiating a SPGSolver object. If no
        # options are selected by the user, default options are used.

        # Keys available:
        #
        # - full_results: return list of criticality values at each iteration
        # 
        # - strategy:
        #   > "monotone": monotone trust region
        #   > "bazfr"   : nonmonotone method according to [1]
        #   > "newfw"   : nonmonotone method according to [2]
        #
        # - gtol: tolerance for detecting convergence on the gradient
        #
        # - maxiter: maximum number of iterations allowed
        #
        # - verbose: verbosity level
        #            0: only convergence info
        #            1: only show time and final stats
        #            2: show outer iterations
        #            3: everything (except debug which is set separately)
        # - changevar: boolean option to allow for a change of variables
        #              before starting the method. Currently disabled
        #              due to bad performance

        super()._setoptions()
        self.options = options
        keys = self.options.keys()

        if "full_results" not in keys:
            self.options["full_results"] = False
        elif type(self.options["full_results"]) != bool:
            raise Exception("full_results must be a boolean")            
        
        if "strategy" not in keys:
            self.options["strategy"] = "newfw"
        elif self.options["strategy"] not in ("monotone", "bazfr", "newfw"):
            raise Exception("strategy not implemented")

        if "gtol" not in keys:
            self.options["gtol"] = 1e-3
        elif type(self.options["gtol"]) != float:
            raise Exception("gtol must be a float")

        if "maxiter" not in keys:
            self.options["maxiter"] = 2000
        elif type(self.options["maxiter"]) != int:
            raise Exception("maxiter must be an integer")

        if "verbose" not in keys:
            self.options["verbose"] = 1
        elif self.options["verbose"] not in (0,1,2,3):
            raise Exception("verbose must be 0, 1, 2 or 3")

        if "changevar" not in keys:
            self.options["changevar"] = False
        elif type(self.options["changevar"]) != bool:
            raise Exception("changevar must be True or False")

    def solve(self, problem):

        """
        Effectively solve the problem using the SPG method.

        Input:

          ``problem``: ``ProcrustesProblem`` instance
           
        Output:

          ``result``: ``OptimizationResult`` instance
        """
        
        fval, normgrad, exitcode, msg = spectral_setup(problem, self.solvername, \
                                                       self.options)

        if self.options["full_results"]:
            if "total_fun" not in problem.stats.keys() or \
               "total_grad" not in problem.stats.keys():
                raise Exception("For full results, set problem.stats[\"total_fun\"]"
                                " and problem.stats[\"total_grad\"]")
            result = OptimizeResult(success=(exitcode == 0),
                                    status=exitcode,
                                    message=msg,
                                    fun=fval,
                                    normgrad=normgrad,
                                    nbiter=problem.stats["nbiter"],
                                    nfev=problem.stats["fev"],
                                    total_fun=problem.stats["total_fun"],
                                    total_grad=problem.stats["total_grad"])
        else:
            result = OptimizeResult(success=(exitcode == 0),
                                    status=exitcode,
                                    message=msg,
                                    fun=fval,
                                    normgrad=normgrad,
                                    nbiter=problem.stats["nbiter"],
                                    nfev=problem.stats["fev"])          

        return result

class GKBSolver(SPGSolver):

    """
    Subclass containing the call to the ``spectral_setup()`` function 
    corresponding to the Spectral Projected Gradient Method using 
    incomplete Golub-Kahan Bidiagonalization (Lanczos) as described in 
    :ref:`[2] <bibliography>`. This class extends the ``SPGSolver`` class,
    with some variation in the input and output parameters.

    Usage example:

       >>> mysolver = skp.GKBSolver(verbose=3)
       >>> result = mysolver.solve(problem)

    Input:

    ``key = value``: keyword arguments available
       - ``full_results``: (*default*: ``False``)
          Return list of criticality values at each iteration (for later
          comparison between solvers)
       - ``strategy``: (*default*: ``"newfw"``)
          - ``"monotone"``: 
             monotone trust region
          - ``"bazfr"`` : 
             nonmonotone method according to :ref:`[1] <bibliography>`
          - ``"newfw"`` : 
             nonmonotone method according to :ref:`[2] <bibliography>`
       - ``gtol``: (*default*: ``1e-3``)
          tolerance for detecting convergence on the gradient
       - ``maxiter``: (*default*: ``2000``)
          maximum number of iterations allowed
       - ``verbose``: (*default*: ``1``)
          verbosity level. Current options:
          - ``0``: only convergence info
          - ``1``: only show time and final stats
          - ``2``: show outer iterations
          - ``3``: everything (except debug which is set separately)
       - ``changevar``: (*default*: ``False``)
          boolean option to allow for a change of variables before starting the 
          method. Currently disabled due to bad performance.

    Output:

    ``solver``: ``ProcrustesSolver`` instance

    .. note::

       Since this subclass extends SPGSolver class, we use 
       ``SPGSolver._setoptions`` directly.
    """

    def __init__(self, **kwargs):
        super().__init__()
        super()._setoptions(options=kwargs)
        self.solvername = "gkb"

    def solve(self, problem):

        """
        Effectively solve the problem using the GKB method.

        Input:

          ``problem``: ``ProcrustesProblem`` instance
           
        Output:

          ``result``: ``OptimizationResult`` instance
        """

        fval, normgrad, exitcode, msg = spectral_setup(problem, self.solvername, self.options)

        if self.options["full_results"]:
            if "total_fun" not in problem.stats.keys() or \
               "total_grad" not in problem.stats.keys():
                raise Exception("For full results, set problem.stats[\"total_fun\"]"
                                " and problem.stats[\"total_grad\"]")
            result = OptimizeResult(success=(exitcode == 0),
                                    status=exitcode,
                                    message=msg,
                                    fun=fval,
                                    normgrad=normgrad,
                                    nbiter=problem.stats["nbiter"],
                                    nfev=problem.stats["fev"],
                                    blocksteps=problem.stats["blocksteps"],
                                    total_fun=problem.stats["total_fun"],
                                    total_grad=problem.stats["total_grad"])
        else:
            result = OptimizeResult(success=(exitcode == 0),
                                    status=exitcode,
                                    message=msg,
                                    fun=fval,
                                    normgrad=normgrad,
                                    nbiter=problem.stats["nbiter"],
                                    nfev=problem.stats["fev"],
                                    blocksteps=problem.stats["blocksteps"])

        return result    

class EBSolver(ProcrustesSolver):

    """
    Subclass containing the call to the ``eb_solver()`` function 
    corresponding to the Expansion-Balance method as described in 
    :ref:`[3] <bibliography>`. 

    Usage example:

       >>> mysolver = skp.EBSolver(verbose=3)
       >>> result = mysolver.solve(problem)

    Input:

    ``key = value``: keyword arguments available
       - ``full_results``: (*default*: ``False``)
          Return list of criticality values at each iteration (for later
          comparison between solvers)
       - ``tol``: (*default*: ``1e-6``)
          tolerance for detecting convergence
       - ``maxiter``: (*default*: ``2000``)
          maximum number of iterations allowed
       - ``verbose``: (*default*: ``1``)
          verbosity level. Current options:
          - ``0``: only convergence info
          - ``1``: only show time and final stats

    Output:

    ``solver``: ``ProcrustesSolver`` instance
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self._setoptions(options=kwargs)
        self.solvername = "eb"

    def solve(self, problem):
        
        """
        Effectively solve the problem using the Expansion-Balance method.

        Input:

          ``problem``: ``ProcrustesProblem`` instance
           
        Output:

          ``result``: ``OptimizationResult`` instance
        """

        fval, exitcode, msg = eb_solver(problem, self.options)

        if self.options["full_results"]:
            if "total_fun" not in problem.stats.keys() or \
               "total_crit" not in problem.stats.keys():
                raise Exception("For full results, set problem.stats[\"total_fun\"]"
                                " and problem.stats[\"total_crit\"]")
            result = OptimizeResult(success=(exitcode == 0),
                                    status=exitcode,
                                    message=msg,
                                    fun=fval,
                                    nbiter=problem.stats["nbiter"],
                                    nfev=problem.stats["fev"],
                                    total_fun=problem.stats["total_fun"],
                                    total_crit=problem.stats["total_crit"])
        else:
            result = OptimizeResult(success=(exitcode == 0),
                                    status=exitcode,
                                    message=msg,
                                    fun=fval,
                                    nbiter=problem.stats["nbiter"],
                                    nfev=problem.stats["fev"])

        return result
    
    def _setoptions(self, options):

        """
        Sets and validates options for the EBSolver.

        *This method should not be called directly; it is called by the 
        EBSolver constructor.*
        """
        
        # Options for the solver.
        # The user has the option of not calling this method explicitly,
        # in which case default options are used. 

        # Keys available:
        #
        # - full_results: return list of criticality values at each iteration
        # 
        # - tol: tolerance for detecting convergence
        #
        # - maxiter: maximum number of iterations allowed
        #
        # - verbose: verbosity level
        #            0: only convergence info
        #            1: only show time and final stats
        #            2: show outer iterations
        #            3: everything (except debug which is set separately)

        # TODO in the future, if we allow non square problems
        # ALSO enable this is tests.
        # if solver == "eb":
        #     # Check if dimensions match.
        #     m,n,p,q = problem.sizes
        #     if m != n or p!=q:
        #         raise Exception("Cannot use rectangular matrices with EB.")
        #     if sp.norm(problem.C-np.eye(p,p), 'inf') > 1e-16:
        #         raise Exception("Error! EB method can only be used if "
        #                         "problem.C is eye(p,p).")

        super()._setoptions()
        self.options = options
        keys = self.options.keys()

        if "full_results" not in keys:
            self.options["full_results"] = False
        elif type(self.options["full_results"]) != bool:
            raise Exception("full_results must be a boolean")            

        if "tol" not in keys:
            self.options["tol"] = 1e-6
        elif type(self.options["tol"]) != float:
            raise Exception("tol must be a float")

        if "maxiter" not in keys:
            self.options["maxiter"] = 2000
        elif type(self.options["maxiter"]) != int:
            raise Exception("maxiter must be an integer")

        if "verbose" not in keys:
            self.options["verbose"] = 1
        elif self.options["verbose"] not in (0,1,2,3):
            raise Exception("verbose must be 0, 1, 2 or 3")

class GPISolver(ProcrustesSolver):

    """
    Subclass containing the call to the ``gpi_solver()`` function 
    corresponding to the Generalized Power Iteration method as described in 
    :ref:`[5] <bibliography>`. 

    Usage example:

       >>> mysolver = skp.GPISolver(verbose=3)
       >>> result = mysolver.solve(problem)

    Input:

    ``key = value``: keyword arguments available
       - ``full_results``: (*default*: ``False``)
          Return list of criticality values at each iteration (for later
          comparison between solvers)
       - ``tol``: (*default*: ``1e-3``)
          tolerance for detecting convergence
       - ``maxiter``: (*default*: ``2000``)
          maximum number of iterations allowed
       - ``verbose``: (*default*: ``1``)
          verbosity level. Current options:
          - ``0``: only convergence info
          - ``1``: only show time and final stats

    Output:

    ``solver``: ``ProcrustesSolver`` instance
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self._setoptions(options=kwargs)
        self.solvername = "gpi"

    def solve(self, problem):
        
        """
        Effectively solve the problem using the Generalized Power Iteration
        method.

        Input:

          ``problem``: ``ProcrustesProblem`` instance
           
        Output:

          ``result``: ``OptimizationResult`` instance
        """

        fval, exitcode, msg = gpi_solver(problem, self.options)

        if self.options["full_results"]:
            if "total_fun" not in problem.stats.keys() or \
               "total_crit" not in problem.stats.keys():
                raise Exception("For full results, set problem.stats[\"total_fun\"]"
                                " and problem.stats[\"total_crit\"]")
            result = OptimizeResult(success=(exitcode == 0),
                                    status=exitcode,
                                    message=msg,
                                    fun=fval,
                                    nbiter=problem.stats["nbiter"],
                                    nfev=problem.stats["fev"],
                                    total_fun=problem.stats["total_fun"],
                                    total_crit=problem.stats["total_crit"])
        else:
            result = OptimizeResult(success=(exitcode == 0),
                                    status=exitcode,
                                    message=msg,
                                    fun=fval,
                                    nbiter=problem.stats["nbiter"],
                                    nfev=problem.stats["fev"])

        return result
    
    def _setoptions(self, options):

        """
        Sets and validates options for the GPISolver.

        *This method should not be called directly; it is called by the 
        GPISolver constructor.*
        """
        
        # Options for the solver.
        # The user has the option of not calling this method explicitly,
        # in which case default options are used. 

        # Keys available:
        #
        # - full_results: return list of criticality values at each iteration
        # 
        # - tol: tolerance for detecting convergence
        #
        # - maxiter: maximum number of iterations allowed
        #
        # - verbose: verbosity level
        #            0: only convergence info
        #            1: only show time and final stats
        #            2: show outer iterations
        #            3: everything (except debug which is set separately)

        super()._setoptions()
        self.options = options
        keys = self.options.keys()

        if "full_results" not in keys:
            self.options["full_results"] = False
        elif type(self.options["full_results"]) != bool:
            raise Exception("full_results must be a boolean")            

        if "tol" not in keys:
            self.options["tol"] = 1e-6
        elif type(self.options["tol"]) != float:
            raise Exception("tol must be a float")

        if "maxiter" not in keys:
            self.options["maxiter"] = 2000
        elif type(self.options["maxiter"]) != int:
            raise Exception("maxiter must be an integer")

        if "verbose" not in keys:
            self.options["verbose"] = 1
        elif self.options["verbose"] not in (0,1,2,3):
            raise Exception("verbose must be 0, 1, 2 or 3")

def spectral_setup(problem, solvername, options):

    """
    Set up parameters according to the optimization method chosen.
    """
    problem.stats["nbiter"] = 0
    problem.stats["svd"] = 0
    problem.stats["fev"] = 0
    problem.stats["gradient"] = 0
    if options["full_results"]:
        problem.stats["total_fun"] = []
        problem.stats["total_grad"] = []

    debug = False
    verbose = options["verbose"]

    m, n, p, q = problem.sizes
    
    f = 0
    exitcode = 0
    normgrad = np.Inf
    residual = normgrad

    # X is the initial guess
    X = np.zeros((n,p))
    U = np.zeros((m,m))
    V = np.zeros((n,n))
    T = np.zeros((m,n+q)) # because of the last block
    
    # Decide whether we are going to solve by blocks or not.
    
    if solvername == "spg":
        U, S, VT = sp.svd(problem.A)
        problem.stats["svd"] = problem.stats["svd"]+1
 
        # 
        # Computing starting point
        # 
        if options["changevar"]:        
            # Change of variables: This is done to improve performance. 
            # Solving this problem is equivalent to solving the original one.
            # THIS IS NOT WORKING. USE CHANGEVAR = FALSE FOR BETTER
            # PERFORMANCE.
            X = np.copy(VT[0:p,0:n].T)
            Aorig = problem.A.copy()
            A = np.zeros((m,n))
            for i in range(0,min(m,n)):
                A[i,i] = S[i]
            Bk = np.dot(U.T, problem.B)
        else:
            X = np.zeros((n,p))
            Bk = np.copy(problem.B)

        exitcode, f, X, normgrad, nbiter, msg = spectral_solver(problem, m, n, \
                                                                X, problem.A, \
                                                                Bk, solvername,\
                                                                options)

        problem.stats["nbiter"] = nbiter
        
        if options["changevar"]:
            # Going back to the original variable
            Xk = np.dot(VT.T, X)
        else:
            Xk = np.copy(X)
            
        R = np.dot(problem.A, np.dot(Xk, problem.C)) - problem.B
        residual = sp.norm(R, 'fro')**2
        
    elif solvername == "gkb":

        problem.stats["blocksteps"] = 0
        
        residuals = []
        # Setting up number of steps allowed in block Lanczos mode
        maxsteps = m/q

        if options["verbose"] == 1:
            print("  outer         f               normg   nbinnerits ")
            print("====================================================")
        # k = current Lanczos block
        k = 1
        while k <= maxsteps and normgrad > options["gtol"]:
            # T = Ak is the partial bidiagonalization of A
            # [U,Ak,V] = bidiag3_block(A,B,q,steps);
            # In blockbidiag, we do a loop with i = partial+1,nsteps;
            # this is the block we are currently working on 

            # Current block starts at (partial+1)*s, ends at nsteps*s
            partial = k-1
            nsteps = k

            # largedim and smalldim are the current dimensions of our problem:
            # A(largedim, smalldim), B(largedim, q), X(smalldim, p)
            largedim = q*(k+1)
            #if largedim < m:
            # Incomplete bidiagonalization:
            # T(q*(k+1),q*k), U(m,q*(k+1)), V(n,q*k)
            smalldim = q*k
            #else:
            # smalldim = m # = n

            if k < maxsteps:
                # B1 is a p by p block used in the computation of the new
                # BLOBOP residual
                
                U, V, T, B1, reorth = blockbidiag(problem, U, V, T, \
                                                  nsteps, partial)

                # Akp1 is ok
                Akp1 = T[largedim-q:largedim, smalldim:smalldim+q]

                if options["verbose"] > 2:
                    print("\n       Finished bidiag: Reorth: {}\n" \
                          .format(reorth))
            else:
                largedim = q*k
                smalldim = q*k
                
            if options["verbose"] > 0:
                print(" ----> GKB Iteration {}: Tk is {}x{}" \
                      .format(k, largedim, smalldim))
          
            # A(m,n) X(n,p) C(p,q) - B(m,q)

            debug = False
            if debug:
                print("\nT = {}\n".format(T[0:largedim, 0:smalldim]))
                print("U = {}\n".format(U[0:m, 0:largedim]))
                print("V = {}\n".format(V[0:n, 0:smalldim]))
                AV = np.dot(problem.A, V[0:n, 0:smalldim])
                prod = np.dot(U[0:m, 0:largedim].T, AV)
                print("T - UT*A*V = {}\n" \
                      .format(T[0:largedim, 0:smalldim] - prod))

            if options["verbose"] > 2:
                AV = np.dot(problem.A, V[0:n, 0:smalldim])
                prod = np.dot(U[0:m, 0:largedim].T, AV)
                print("       MaxError = {}\n" \
                      .format(np.max(T[0:largedim,0:smalldim] - prod)))

            # Bk(q*(k+1),q) = U(m,q*(k+1))'*B(m,q)
            Bk = np.dot(U[0:m, 0:largedim].T, problem.B)
            
            # T(q*(k+1),q*k) X(q*k,p) C(p,q) - Bk(q*(k+1),q)
            
            if k == 1:
                X[0:smalldim,0:p] = np.copy(V[0:smalldim,0:p])
            else:
                # X(1:q*(k-1),1:p) = result from last run
                X[q*(k-1):smalldim,0:p] = np.zeros((smalldim-q*(k-1), p))

            VT = np.zeros((n,n))
            VT[0:smalldim,0:n] = np.copy(V[0:n,0:smalldim].T)

            # temp is old normdummyg
            exitcode, f, X[0:smalldim, 0:p], temp, outer, msg \
                = spectral_solver(problem, largedim, smalldim, \
                                  X[0:smalldim,0:p], \
                                  T[0:largedim,0:smalldim], \
                                  Bk[0:largedim,0:q], \
                                  solvername, options)

            problem.stats["nbiter"] = problem.stats["nbiter"] \
                                      + (largedim/m)*outer
            
            Yk = np.copy(X[0:smalldim, 0:p])
            Xk = np.dot(V[0:n,0:smalldim], Yk)
            
            R = np.dot(problem.A, np.dot(Xk, problem.C)) - problem.B
            residual = sp.norm(R, 'fro')**2
            residuals.append(residual)
            
            grad = 2.0*np.dot(problem.A.T, np.dot(R, problem.C.T))
            
            gradproj = np.dot(Xk, np.dot(Xk.T, grad) + np.dot(grad.T,Xk)) \
                       - 2.0*grad
            normgrad = sp.norm(gradproj, 'fro')

            if options["verbose"] > 1:
                print("\n       Gradient norm       = {}" \
                      .format(normgrad))
                print("       Residual norm       = {}\n"   \
                      .format(residual))

            ###################################### BLOBOP
            if k < maxsteps:
                calB = np.zeros((largedim,p))
                calB[0:p,0:p] = B1
                Tk = T[0:largedim, 0:smalldim]
                res1 = np.dot((np.eye(smalldim,smalldim) - np.dot(Yk,Yk.T)), \
                              np.dot(Tk.T, np.dot(Tk,Yk) - calB))
                resBlobop1 = sp.norm(res1, 'fro')
            
                # Z(p)(k) is the last pxp block of Yk.
                Zpk = np.copy(Yk[(k-1)*p:smalldim, 0:p])
                Bkp1 = T[largedim-p:largedim, smalldim-p:smalldim]
                Vk = V[0:n, 0:smalldim]
                
                prod1 = np.eye(n,n) - np.dot(Vk, \
                                             np.dot(Yk, np.dot(Yk.T, Vk.T)))
                prod2 = np.dot(V[0:n, smalldim:smalldim+p], \
                               np.dot(Akp1, np.dot(Bkp1, Zpk)))
                res2 = np.dot(prod1, prod2)
                resBlobop2 = sp.norm(res2, "fro")
                
                newResidual = np.sqrt(resBlobop1**2 + resBlobop2**2)
                if options["verbose"] > 1:
                    print("       New BLOBOP Residual = {}\n".format(newResidual))
            ###################################### BLOBOP
            
            k = k + 1
            
        # end while
        
        problem.stats["blocksteps"] = k-1

    else:
        # STILL TO BE DEVELOPED.
        print("\nWRONG SOLVER!\n")

    if normgrad <= options["gtol"]:
        msg = _status_message['success']
        if options["verbose"] > 0:
            print(msg)

    return f, normgrad, exitcode, msg

def spectral_solver(problem, largedim, smalldim, X, A, B, solvername, options):

    """ 
    Nonmonotone Spectral Projected Gradient solver for problems of the type

       .. math::

          \\min \\lVert AXC - B\\rVert_F^2  \\qquad s.t. X^TX = I

    The method is described in references [1] and [2], and we implement a few
    variations (including a monotone version, a nonmonotone version using the 
    strategy described in [1], and a nonmonotone version using the strategy 
    described in [2]; check below for more details on how to select these 
    different algorithms).

    This function is called by ``spectral_solver`` from both GKBSolver and
    SPGSolver, with different parameters.
    
    Input:

    - ``problem``: ``ProcrustesProblem`` instance
    - ``largedim``: ``int``
    - ``smalldim``: ``int``
       Since this function is called by ``spectral_solver``, it is possible
       we are solving a smaller version of the original problem (when using
       GKBSolver, for instance). Thus, ``lagedim`` and ``smalldim`` are the
       dimensions of the current problem being solved by ``spectral_solver``.
    - ``X``: ``ndarray(smalldim, p)``
       Initial guess for the solution X of the Procrustes Problem being solved.
    - ``A``: ``ndarray(largedim, smalldim)``
    - ``B``: ``ndarray(largedim, q)``
    - ``solvername``: str
       Takes values ``spg`` or ``gkb`` (used to decide if ``full_results`` can 
       be reported).
    - ``options``: ``dict``
       Solver options. Keys available are:
          - ``maxiter``: ``int``
             Maximum number of iterations allowed
          - ``strategy``: ``str``
             ``monot`` (Monotone strategy), ``bazfr`` (Nonmonotone strategy 
             described in [1]) or ``newfw`` (Nonmonotone strategy described 
             in [2])
          - ``verbose``: ``int``
             Can take values in (0,1,2,3)
          - ``gtol``: ``float``
             Tolerance for convergence.
        
    Output:

       - ``exitcode``: ``int``
          0 (success) or 1 (failure)
       - ``f``: ``float``
          Value of the objective function at final iterate
       - ``X``: ``ndarray(smalldim, p)``
          Approximate solution (final iterate)
       - ``normg``: ``float``
          Criticality measure at final iterate
       - ``outer``: ``int``
          Final number of outer iterations performed.

    References:

    [1] J.B. Francisco, F.S. Viloche Bazán, Nonmonotone algorithm for 
        minimization on closed sets with applications to minimization on 
        Stiefel manifolds, Journal of Computational and Applied Mathematics, 
        2012, 236(10): 2717--2727 <http://dx.doi.org/10.1016/j.cam.2012.01.014>
    [2] J.B. Francisco, F.S. Viloche Bazán and M. Weber Mendonça, Non-monotone 
        algorithm for minimization on arbitrary domains with applications to 
        large-scale orthogonal Procrustes problem, Appl. Num. Math., 2017, 
        112: 51--64 <https://doi.org/10.1016/j.apnum.2016.09.018>
    """
    
    # Setup

    # A(largedim, smalldim)
    # B(largedim, q)
    # problem.C(p, q)
    # X(smalldim, p)
    
    m, n, p, q = problem.sizes # original sizes, not reduced
    
    # chi determines by which factor is rho increased at each 
    # unsuccessful iteration
    chi = 5.0 # juliano
    
    #cost = [None]*(options["maxiter"]+1)
    cost = []
    
    # "total_fun" and "total_grad" store the criticality info for each iteration
    if options["full_results"] and solvername == "spg":
        problem.stats["total_fun"] = []
        problem.stats["total_grad"] = []

    # Barzilai-Borwein parameter
    sigma_min = 1.0e-10
    
    # Sufficient decrease parameter for trust region
    # (trratio must be larger than beta1)
    beta1 = 1.0e-10 # beta1 = 1.0e-4_wp, beta1 = 0.5_wp
    
    # memory is the nonmonotone parameter, used to determine how many
    # iterations will be used in the BAZFR strategy to compare the current
    # objective function 
    # with past values.
    if options["strategy"] == "monot":
        memory = 1
    else:
        memory = 10

    # Lu approximates the Lipschitz constant for the gradient of f
    Lu = 2.0*sp.norm(np.dot(problem.C, problem.C.T), 'fro') \
         *sp.norm(np.dot(A.T, A), 'fro')

    if options["verbose"] > 2:
        print("\n          Lu = {}\n".format(Lu))
        
    # R is the residual, norm(R,fro) is the cost.
    # R = A*X*C-B
    R = np.dot(A, np.dot(X, problem.C)) - B

    cost.append(sp.norm(R, 'fro')**2)

    #problem.stats["fev"] = problem.stats["fev"] + 1
    problem.stats["fev"] = problem.stats["fev"] + (largedim/m)

    if options["strategy"] == "newfw":
        eta = 0.2
        quot = 1.0
    f = cost[0]

    # Test optimality of X0
    grad, normg = optimality(A, problem.C, X, R)

    problem.stats["gradient"] = problem.stats["gradient"] + 1
    if options["full_results"] and solvername == "spg":
        problem.stats["total_fun"].append(f)
        problem.stats["total_grad"].append(normg)
    
    if options["verbose"] > 1:
        print("\n          OUTER ITERATION 0:\n")
        print("             f = {}".format(f))
        print("             normg = {}".format(normg))
    elif options["verbose"] == 1:
        print("  nbiter         f              cost             normg")
        print("===========================================================")
        print(" {0:>4} {1:>16.4e} {2:>16.4e} {3:>16.4e}".format(0, f, f, normg))        
       
    #problem.stats["nbiter"] = 0
    outer = 0

    # If flag_while, then continue cycling.
    flag_while = True
    oldbigres = 0.0
    ftrial = 0.0

    while normg > options["gtol"] and flag_while \
          and outer < options["maxiter"]:
        
        # Computation of the trust-region step parameters
        if outer == 0:
            sigma_bb = 0.5
        else:
            step = np.copy(X - Xold)
            AstepC = np.dot(A, np.dot(step, problem.C))            
            sigma_bb = (sp.norm(AstepC, 'fro')**2)/(sp.norm(step, 'fro')**2)

        # sigma_bb is the Barzilai-Borwein parameter.
        sigma = max(sigma_min, sigma_bb)
        
        trratio = beta1/2.0

        # rho is the regularization parameter for the quadratic model
        rho = sigma
        if options["verbose"] > 2:
            print("             sigma = rho = {}\n".format(sigma))
        nbinnerit = 0
            
        # Inner iteration
        # =============================================================

        while flag_while and trratio < beta1:
          
            if options["verbose"] > 2:
                print("\n             INNER ITERATION {}:\n".format(nbinnerit))
                print("             f = {}\n".format(cost[outer]))
                print("             normg = {}\n".format(normg))

            # Solving the subproblem: Xtrial, the solution to the subproblem,
            # is defined as 
            # Xtrial = U*V' = UW*VWT
            # where
            # [U,S,V] = svd(W,0)
            # where the above line is the "economy size" svd decomposition
            # of W, defined as
            
            W = np.copy(X - (1.0/(rho + sigma))*grad)

            # If X is m-by-n with m > n, then svd(X,0) computes only the first
            # n columns of U and S is (n,n)

            UW, SW, VWT = sp.svd(W, full_matrices=False)
            #UW, SW, VWT = sp.svd(W)

            # W(smalldim,p)
            # UW(smalldim,min(smalldim,p))
            # VWT(min(smalldim,p),p)

            Xtrial = np.dot(UW, VWT)
            problem.stats["svd"] = problem.stats["svd"] + 1

            # Computing constraint violation to see if the subproblem
            # solution has been satisfactorily solved
            constraintviolation = np.abs(sp.norm(np.dot(Xtrial.T, Xtrial), \
                                                 np.inf) - 1.0)

            if constraintviolation >= 1.0e-5:
                msg = _status_message['infeasible']
                print('Warning: ' + msg)
                return
        
            # Rtrial = A*Xtrial*C - B
            Rtrial = np.dot(A, np.dot(Xtrial, problem.C)) - B
            
            # ftrial = norm(Rtrial, fro)**2
            ftrial = sp.norm(Rtrial, 'fro')**2
                
            #problem.stats["fev"] = problem.stats["fev"]+1
            problem.stats["fev"] = problem.stats["fev"] + (largedim/m)
        
            if options["verbose"] > 2:
                print("             ftrial = {}\n".format(ftrial))

            ared = f - ftrial
            pred = - np.trace(np.dot(grad.T, (Xtrial-X)) \
                              - (sigma/2.0)*sp.norm(Xtrial-X, 'fro')**2)
            
            if np.abs(pred) < 1.0e-15:
                msg = _status_message['smallpred']
                print('Warning: ' + msg)
                trratio = 0.0
                flag_while = False
            else:
                trratio = ared/pred
            
            if pred < 0:
                msg = _status_message['negativepred']
                print('Warning: ' + msg)
                flag_while = False
                # trratio = 0.0
                
            if options["verbose"] > 2:
                print("             ared = {}\n".format(ared))
                print("             pred = {}\n".format(pred))
                print("             trratio = {}\n".format(trratio))
                    
            if trratio > beta1:
                if options["verbose"] > 2:
                    print("\n             INNER ITERATION FINISHED: success\n")
                    print("             trratio = {}\n".format(trratio))
                    print("             beta1 = {}\n".format(beta1))
            
            # Below is equation (15) from (Francisco, Bazan, 2012)
            if flag_while:
                rho = chi*rho
                if rho > Lu:
                    if options["verbose"] > 1:
                        print("             WARNING: Using Lu "
                              " parameter = {} to ensure sufficient decrease "
                              " (inner {} and outer = {})\n"\
                              .format(Lu, nbinnerit, outer))
                    sigma = Lu
                
                if options["verbose"] > 2:
                    print("             rho = {}\n".format(rho))
                    print("             sigma = {}\n".format(sigma))
            
                nbinnerit = nbinnerit + 1
            
                if nbinnerit >= options["maxiter"]:
                    msg = _status_message['maxiter']
                    print('Warning: ' + msg + '(inner loop)')
                    trratio = 1.0 # just to leave the while
                    
        # end while innerit ================================================

        Xold = X.copy()
        X = Xtrial.copy()
        R = Rtrial.copy()
        
        cost.append(ftrial)
        # TODO: fix this (refactor?)
        # Compute cost
        if options["strategy"] == "MONOT":
            f = cost[outer+1]
        elif options["strategy"] == "BAZFR":
            if outer < memory:
                f = max(cost)
            else:
                f = max(cost[outer+1-memory:outer+1])
        else:
            # newfw
            qold = quot
            quot = eta*qold + 1.0
            f = (eta*qold*f+ftrial)/quot
                  
        grad, normg = optimality(A, problem.C, X, R)
        problem.stats["gradient"] = problem.stats["gradient"] + 1
        if options["full_results"] and solvername == "spg":
            problem.stats["total_fun"].append(cost[outer+1])
            problem.stats["total_grad"].append(normg)

        if options["verbose"] > 2:
            print("\n          OUTER ITERATION {}:\n" \
                  .format(outer+1))
            print("             f = {}".format(f))
            print("             normg = {}".format(normg))
        elif options["verbose"] == 1:
            # outer f normg innerits
            print(" {0:>4} {1:>16.4e} {2:>16.4e} {3:>16.4e} {4:>4}".format(outer+1, \
                                                                f, \
                                                                cost[outer+1], \
                                                                normg,\
                                                                nbinnerit))
            
        outer = outer+1

    #end while 

    if outer >= options["maxiter"]:
        msg = _status_message["maxiter"]
        exitcode = 1
        print('Warning: ' + msg)
    else:
        exitcode = 0
        msg = ""

    # Since we already updated nbiter, f is now cost[outer] instead of
    # cost[outer+1]
    
    f = cost[outer]
    
    return exitcode, f, X, normg, outer, msg
    
def blockbidiag(problem, U, V, T, steps, partial):
    
    """ 
    This function builds block matrices U, V orthogonal such that
    U'*A*V=T, where T is bidiagonal.
    A(m,n), T(m,n), U(m,m), V(n,n), B(m,s)
    """

    m, n, s, q = problem.sizes
    
    # The current Lanczos step goes from (partial+1)*s:steps*s
    
    # Tolerance for reorthogonalization
    gstol = 1.0e-8
    
    debug = False
    # steps = min(m,n)/s-1
    reorth = 0

    # First, we will compute the QR decomposition of B to find
    # U[:,0:s]*B1[0:s,0:s] = B
    BB = problem.B.copy()

    # scipy.linalg.qr(a, overwrite_a=False, lwork=None, mode='full',
    #                 pivoting=False, check_finite=True)
    # Calculate the decomposition A = Q R where Q is unitary/orthogonal and R
    # upper triangular.
    # Parameters:
    #  - a : (M, N) array_like -> Matrix to be decomposed
    #  - overwrite_a : bool, optional -> Whether data in a is overwritten (may
    #                                   improve performance)
    #  - mode : {‘full’, ‘r’, ‘economic’, ‘raw’}, optional -> Determines what
    #                                   information is to be returned: either
    #                                   both Q and R (‘full’, default), only R
    #                                   (‘r’) or both Q and R but computed in
    #                                   economy-size (‘economic’). The final
    #                                   option ‘raw’ (added in Scipy 0.11) makes
    #                                   the function return two matrices
    #                                   (Q, TAU) in the internal format used
    #                                   by LAPACK.
    # Returns:
    #  - Q : float or complex ndarray -> Of shape (M, M), or (M, K) for
    #                                   mode='economic'. Not returned if
    #                                   mode='r'.
    #  - R : float or complex ndarray -> Of shape (M, N), or (K, N) for
    #                                   mode='economic'. K = min(M, N).

    # R.shape = BB.shape (m,q)
    Q, R = sp.qr(BB, overwrite_a=True, mode='economic')
    B1 = np.copy(R[0:s,0:s])
    # (obs: B1[0:s,0:s] does not go into T!)
    # B1 is needed for the new residual computation of BLOBOP
    
    if partial == 0:
        # This is the first block
        U[0:m,0:s] = np.copy(Q[0:m,0:s])

        # Now, we compute the QR decomposition of A'*U(:,1:s)
        # to find V(:,1:s)T(1:s,1:s) = A'*U(:,1:s)

        Q, R = sp.qr(np.dot(problem.A.T, U[0:m,0:s]), overwrite_a=True, \
                     mode='economic')
        V[0:n,0:s] = np.copy(Q[0:n,0:s])

        # The first block of T is A{1}.T, which is R above.
        T[0:s,0:s] = np.copy(R[0:s,0:s].T)

    # Main loop
    # i indicates how many blocks have been used.
    for i in range(partial+1, steps+1):
        # i indicates what block row we are computing (starting from 2, since A1 has been computed above)
        inds = s*i

        # U{i}   = U[:, inds-s:inds]
        # A{i}.T = T[inds-s:inds, inds-s:inds]
        # V{i}   = V[:, inds-s:inds]

        # In the original algorithm, we used
        # U_{i+1}B_{i+1} = A V{i} - U{i} A{i}.T = prod[0:m,0:s] (reduced QR)
        # However, in our case we will choose to compute U{i+1} and B{i+1}
        # without the QR decomposition; we will instead use a complete
        # reorthogonalization to build these matrices.

        # First, we take the matrix that should be decomposed into Q and R:
        prod = np.dot(problem.A, V[0:n, inds-s:inds]) \
               - np.dot(U[0:m, inds-s:inds], T[inds-s:inds, inds-s:inds])
        # prod is (m,s)

        # Next, we apply the modified Gram-Schmidt algorithm with complete
        # reorthogonalization if needed to find Uip1 and Bip1 such that
        # Uip1Bip1 = QR(prod)
        Umat = np.copy(U)
        Uip1, Bip1, reorth = bidiaggs(inds, prod, Umat, gstol, reorth)

        if debug:
            print("Erro bidiaggs = {}\n" \
                  .format(sp.norm(np.dot(Uip1[0:m,inds:inds+s], Bip1) - prod)))
        
        # Now, the blocks go into U and T.
        U = np.copy(Uip1)
        T[inds:inds+s, inds-s:inds] = np.copy(Bip1)

        # Same for V:
        # First, we take the matrix that should be decomposed into Q and R:
        # V_{i+1}A{i+1} = A'*U_{i+1} - V_iB_{i+1}' = prod[0:n,0:s]         
        prod = np.dot(problem.A.T, U[0:m, inds:inds+s]) \
               - np.dot(V[0:n, inds-s:inds], T[inds:inds+s, inds-s:inds].T)

        Vmat = np.copy(V)
        Vip1, Aip1, reorth = bidiaggs(inds, prod, Vmat, gstol, reorth)
        V = np.copy(Vip1)
        T[inds:inds+s, inds:inds+s] = np.copy(Aip1.T)

        # TODO: This should be a test!
        if debug and options["verbose"] > 1:
            debug_bidiag(i, s, inds, A, B, U, V, T)
            
    if debug and options["verbose"] > 1:
        print("\n        MaxError: max(T-U'*A*V) = {}\n" \
              .format(np.max(T[0:m,0:n] - np.dot(U.T,np.dot(problem.A,V)))))
        print("\n        max(V'*V - I) = {}\n" \
              .format(np.max(np.dot(V.T, V) - np.eye(n,n))))
        print("\n        max(U'*U - I) = {}\n" \
              .format(np.max(np.dot(U.T, U) - np.eye(m,m))))
        
    # Output matrices: U(m,m), V(n,n), T(m,n), B1(s,s)
    return U, V, T, B1, reorth

def bidiaggs(inds, prod, mat, gstol, reorth):

    """ Computing one block of blockbidiag at a time: 
    this routine computes the QR decomposition of A using 
    the modified Gram-Schmidt algorithm with reorthogonalizations. """
    
    s = prod.shape[1]
    R = np.zeros((s,s))
    A = np.copy(prod)

    # This routine can be called twice inside blockbidiag:
    # - If mat is Umat, then mat is (m,m) and prod is (m,s)
    # - If mat is Vmat, then mat is (n,n) and prod is (n,s)
    # We are computing a submatrix of mat which is Q for prod = QR
    # However, we need the complete mat here because we will
    # reorthogonalize, when necessary, against all columns of Umat.
    
    # Now, we will reorthogonalize each column k of prod with respect to all
    # previous columns
    for k in range(0, s):
        # k is the current column
        indspk = inds+k

        # Reorthogonalization with respect to all previous columns of U:
        # the code below is equivalent to
        # for j in range(0, indspk):
        #    prod[:, k] = prod[:, k] - np.dot(prod[:, k], mat[:, j]) * mat[:, j]
        # We have rearranged so that
        #
        # prod[:,k] = prod[:,k] - sum(mat*diag(prod[:,k]'*mat))
        #
        # where the sum is done so that the result is a column vector.
        # (the transpose on prod is irrelevant; np.dot takes care of that)
        temp = np.diag(np.dot(A[:, k].T, mat[:, 0:indspk]))
        A[:, k] = A[:, k] - np.sum(np.dot(mat[:, 0:indspk], temp), axis=1)
        # If mat = Umat, then
        # B_{i+1} = T(s*i+1:s*(i+1),s*(i-1)+1:s*i)
        # B_{i+1}(k,k) = norm(UU_{i+1}(:,k))

        # If mat = Vmat, then
        # A_{i+1} = T(inds+1:inds+s,inds+1:inds:s)
        # A_{i+1}(k,k) = norm(VV_{i+1}(:,k))

        # B{i+1}(k, k) = norm(prod[0:m, k])
        # A{i+1}[k, k] = sp.norm(prod[0:n,k])
        R[k, k] = sp.norm(A[:, k])

        if abs(R[k, k]) < gstol:
            reorth = reorth + 1
            # TODO CHECK THIS
            # Trying out not using random numbers when
            # reorthogonalizing to keep results controlled
            #A[:,k] = np.zeros((A.shape[0],))
            #A[k,k] = 1.0
            for j in range(0, A.shape[0]):
                A[j, k] = np.random.randn(1)

            # Reorthogonalize against all computed elements
            temp = np.diag(np.dot(A[:, k].T, mat[:, 0:indspk]))
            A[:, k] = A[:, k] - np.sum(np.dot(mat[:, 0:indspk], temp), axis=1)

            mat[:, indspk] = A[:, k] / sp.norm(A[:, k])
        else:
            # U_{i+1}(:,k) = UU_{i+1}(:,k)/B_{i+1}(k,k)
            # V_{i+1}(:,k) = VV_{i+1}(:,k)/A_{i+1}(k,k)
            mat[:, indspk] = A[:, k] / R[k, k]

        for j in range(k + 1, s):
            # B_{i+1}(k,j) = U_{i+1}(:,k)'*UU_{i+1}(:,j)
            # A_{i+1}(k,j) = V_{i+1}(:,k)'*VV_{i+1}(:,j)
            R[k, j] = np.dot(mat[:, indspk], A[:, j])
            # UU_{i+1}(:,j) = UU_{i+1}(:,j) - U_{i+1}(:,k)B_{i+1}(k,j)
            # VV_{i+1}(:,j) = VV_{i+1}(:,j) - V_{i+1}(:,k)A_{i+1}(k,j)
            # prod[0:m, j] = prod[0:m, j] - T[s*i+k, s*(i-1)+j]*U[0:m, s*i+k]
            # prod[0:n, j] = prod[0:n, j] - T[inds+j, indspk]*V[0:n, indspk]

    return mat, R, reorth

#TODO move this to tests? how?
def debug_bidiag(i, s, inds, A, B, U, V, T):
    print("\n       ********* DEBUGGING BLOCKBIDIAG: ************\n")
    # We will check the recurrence relations listed in Karimi, Toutounian 
    print("\n        Iteration i = {}, inds = {}\n".format(i, inds))
    E1 = np.zeros((inds+s, s))
    E1[0:s, :] = np.eye(s,s)
    errorRecurrence1 = sp.norm(B-np.dot(U[:,0:inds+s], np.dot(E1, B1)))
    print("\n        B - UU(i+1)*E1*B1 = {}\n".format(errorRecurrence1))
    #
    # AVk = Ukp1Tk
    errorRecurrence2 = sp.norm(np.dot(A, V[:, 0:inds]) - np.dot(U[:, 0:inds+s], T[0:inds+s, 0:inds]))
    print("\n        A*VV(i) - UU(i+1)T(i) = {}\n".format(errorRecurrence2))
    #
    # ATUkp1 = VkTkT + Vkp1Akp1Ekp1T
    Eip1 = np.zeros((inds+s, s))
    Eip1[inds:inds+s, :] = np.eye(s,s)
    errorRecurrence3 = sp.norm(np.dot(A.T, U[:, 0:inds+s]) - np.dot(V[:, 0:inds], T[0:inds+s, 0:inds].T) - np.dot(V[:, inds:inds+s], np.dot(Aip1, Eip1.T)))
    print("\n        A.T*UU(i+1) - VV(i)*T(i).T - V(i+1)*A(i+1)*E(i+1).T = {}\n".format(errorRecurrence3))

def optimality(A, C, X, R):
    # Test optimality of X0
    # grad = 2*A'*R*C'
    grad = 2.0*np.dot(A.T, np.dot(R, C.T))
    # grad_proj = X* (X'*grad + grad'*X) - 2*grad
    gradproj = np.dot(X, (np.dot(X.T, grad)+np.dot(grad.T, X))) - 2.0*grad
    normg = sp.norm(gradproj, 'fro')
    return grad, normg

def eb_solver(problem, options):
    
    """
    Expansion-Balance solver as presented in [1]

    Here we consider always :math:`m=n`, :math:`p=q`, :math:`C=I`.
    Thus the problem has to be
    
       .. math::
     
          \\min \\lVert A_{n\\times n}X_{n\\times p}-B_{n\\times p}\\rVert_F^2 
          \\qquad s.t. X^TX=I_{p\\times p}    
    
    References:
    
    [1] Zhang, Du - Successive projection method for solving the
                    unbalanced Procrustes problem
    [2] Green B. F., Goers J. C. - A problem with Congruence. The Annual
                    Meeting of the Psychometric Society, Monterey,
                    California 1979.
    [3] Ten Berge J. M. F., Konl D. L. - Orthogonal rotations to maximal
                    agreement for two of more matrices of different column
                    orders. Psychometrica, 1984, 49:49-55.
    """

    problem.stats["nbiter"] = 0
    problem.stats["fev"] = 0
    problem.stats["svd"] = 0
    if options["full_results"]:
        problem.stats["total_fun"] = []
        problem.stats["total_crit"] = []

    m, n, p, q = problem.sizes
    exitcode = 0
    msg = ''

    # Initialization (X = G)
    # From [1], p. 973:
    # "The initial guess G(0) can be a solution to the balance problem
    # min norm(AG-[B, Bhat], 'fro') s. t. G'*G=I
    # with an expansion [B, Bhat] of B. In ref. [2], Bhat was simply set to
    # be zero or a randomly chosen matrix. A better initial guess
    # Bhat = AE
    # was suggested in ref. [3] with E the eigenvector matrix of A'*A
    # corresponding to its n-k smallest eigenvalues."
        
    #G(n,n) = [X(n,p), H(n,n-p)]

    Bhat = np.zeros((n,n-p))

    B = np.concatenate((problem.B, Bhat), axis=1)

    # Find the SVD of A.T*B = USV.T, and define G(0) = U*V.T
    U, S, VT = sp.svd(np.dot(problem.A.T, B))
    problem.stats["svd"] = problem.stats["svd"] + 1
    G = np.dot(U, VT)
    # X = np.copy(G[0:n, 0:p])
    X = np.zeros((n,p))
    
    f = sp.norm(np.dot(problem.A, X) - problem.B, 'fro')**2
    problem.stats["fev"] = problem.stats["fev"] + 1
    if options["full_results"]:
        problem.stats["total_fun"].append(f)
        problem.stats["total_crit"].append(f)
        
    if options["verbose"] > 0:
        print("                     EB Solver")
        print("  nbiter         f             fold-f          tol*fold")
        print("===========================================================")
        print(" {0:>4} {1:>16.4e}".format(0, f))

    criticality = False
    nbiter = 0
    while not criticality and nbiter < options["maxiter"]:

        # Solve the expansion problem
        # min norm(AG-[B, AH], 'fro') s.t. G'G=I
        # by finding the svd of A'[B, AH].
        H = G[0:n, p:n]
        AH = np.dot(problem.A, H)
        B = np.concatenate((problem.B, AH), axis=1)
            
        # Find the SVD of A.T*B = USV.T, and define G = U*V.T
        U, S, VT = sp.svd(np.dot(problem.A.T, B))
        problem.stats["svd"] = problem.stats["svd"]+1
        G = np.dot(U, VT)
        X = np.copy(G[0:n, 0:p])
        #X = 0*G[0:n, 0:p]

        fold = f
        f = sp.norm(np.dot(problem.A, X) - problem.B, 'fro')**2
        problem.stats["fev"] = problem.stats["fev"]+1
        
        # Check for convergence
        criticality = (np.abs(fold - f) < options["tol"]*fold) or \
                      (np.abs(f) < options["tol"]) 

        if options["full_results"]:
            problem.stats["total_fun"].append(f)
            problem.stats["total_crit"].append(min(np.abs(fold-f)/np.abs(fold), \
                                                   np.abs(fold)))

        # Print and loop back
        nbiter = nbiter + 1
        
        if options["verbose"] > 0:
            print(" {0:>4} {1:>16.4e} {2:>16.4e} {3:>16.4e}".format(nbiter, f, fold-f, options["tol"]*fold))
                
    # ===================================================== end while

    if nbiter >= options["maxiter"]:
        msg = _status_message["maxiter"]
        exitcode = 1
        print('Warning: ' + msg)
    else:
        exitcode = 0
        msg = _status_message["success"]
       
    problem.stats["nbiter"] = nbiter
            
    return f, exitcode, msg

def gpi_solver(problem, options):

    """
    Generalized Power Iteration solver as presented in [1]

    Here we consider always C=I.
    Thus the problem has to be

       .. math::
     
          \\min \\lVert A_{m\\times n}X_{n\\times p}-B_{m\\times p}\\rVert_F^2 
          \\qquad s.t. X^TX=I_{p\\times p}    

    References:
    
    [1] F. Nie, R. Zhang, X. Li, A generalized power iteration method for
        solving quadratic problem on the Stiefel manifold, Sci. China Inf. Sci.,
        2017, 60: 112101:1--112101:10.
        <http://dx.doi.org/10.1007/s11432-016-9021-9>
    """

    problem.stats["nbiter"] = 0
    problem.stats["fev"] = 0
    problem.stats["svd"] = 0
    if options["full_results"]:
        problem.stats["total_fun"] = []
        problem.stats["total_crit"] = []

    m, n, p, q = problem.sizes
    exitcode = 0
    msg = ''

    # Initialization (X = I)
    X = np.zeros((n,p))
    #X[0:p,0:p] = np.eye(p,p)
    #X = problem.Xsol

    E = np.dot(problem.A.T, problem.A)
    # gamma is a constant times the largest eigenvalue of E, such that
    # gamma*I - E is positive definite.
    vals = spl.eigs(E, k=1, return_eigenvectors=False)
    gamma = vals[0]
    H = 2*(gamma*np.eye(n,n) - E)
    ATB = 2*np.dot(problem.A.T, problem.B)

    f = sp.norm(np.dot(problem.A, X) - problem.B, 'fro')**2
    problem.stats["fev"] = problem.stats["fev"]+1
    if options["full_results"]:
        problem.stats["total_fun"].append(f)
        problem.stats["total_crit"].append(f)

    if options["verbose"] > 0:
        print("               GPI Solver ")
        print("  nbiter         f             fold-f          ")
        print("===================================================")
        print(" {0:>4} {1:>16.4e}".format(0, f))

    criticality = False
    nbiter = 0
    while not criticality and nbiter < options["maxiter"]:

        M = np.dot(H, X) + ATB
        
        # Find the SVD of M
        U, S, VT = sp.svd(M, full_matrices=False)
        problem.stats["svd"] = problem.stats["svd"]+1
        X = np.dot(U, VT)

        fold = f
        f = sp.norm(np.dot(problem.A, X) - problem.B, 'fro')**2
        problem.stats["fev"] = problem.stats["fev"]+1
        
        # Check for convergence
        criticality = (np.abs(fold - f) < options["tol"]) or (np.abs(f) < options["tol"])

        if options["full_results"]:
            problem.stats["total_fun"].append(f)
            problem.stats["total_crit"].append(min(np.abs(fold-f), np.abs(fold)))
        
        # Print and loop back
        nbiter = nbiter + 1
        
        if options["verbose"] > 0:
            print(" {0:>4} {1:>16.4e} {2:>16.4e}".format(nbiter, f, fold-f))
                
    # ===================================================== end while

    if nbiter >= options["maxiter"]:
        msg = _status_message["maxiter"]
        exitcode = 1
        print('Warning: ' + msg)
    else:
        exitcode = 0
        msg = _status_message["success"]
       
    problem.stats["nbiter"] = nbiter
            
    return f, exitcode, msg

def compare_solvers(problem, *args, plot=False):
    """
    This is a tool to compare different solvers (or the same solver function
    with different parameters) on a given ``ProcrustesProblem`` instance.

    Input:

       - ``problem``: ``ProcrustesProblem`` object
          The problem for which we want to compare solvers.
       - ``args``: ``ProcrustesSolver`` objects
          The solvers we want to compare.

    Output:

       **Statistics and graphs showing a comparison between solvers.**

    .. note::
    
       GKBSolver cannot be compared with the other solvers because of the 
       nature of its iterations. In the future, it should be possible to add 
       a tool to compare it to other algorithms.
    """

    # TODO turn off verbose for comparison

    results = []
    if plot:
        fig, ax = plt.subplots()
        
    for solver in args:
        if solver.solvername == "spg":
            plotlabel = "SPG Solver"
        elif solver.solvername == "eb":
            plotlabel = "EB Solver"
        elif solver.solvername == "gpi":
            plotlabel = "GPI Solver"
        results.append(solver.solve(problem))
        y = np.asarray(results[-1].total_fun)
        plt.semilogy(y, label=plotlabel)
        legend = ax.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Objective")
        plt.title("Problem "+str(problem.problemnumber))
        
    plt.show()
        
    return results
