import numpy as np
from numpy.testing import (assert_raises, assert_allclose, assert_equal,
                           assert_, TestCase, run_module_suite, dec,
                           assert_almost_equal, assert_warns, assert_array_less)
from scipy import linalg as sp
import skprocrustes as skp

# Testing functions inside ProcrustesProblem class:
class TestSetProblem(TestCase):

    #def _setproblem(self, matrices, problemnumber):
    # A(m,n), B(m,q), C(p,q) -> problem matrices

    def test_setproblem_dimensions_square(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10,2,5,10), problemnumber=1)

    def test_setproblem_dimensions(self):
        problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)
        assert_equal(problem.A.shape[0], 10)
        assert_equal(problem.A.shape[1], 10)
        assert_equal(problem.B.shape[0], 10)
        assert_equal(problem.B.shape[1], 2)
        assert_equal(problem.C.shape[0], 2)
        assert_equal(problem.C.shape[1], 2)

        problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=2)
        assert_equal(problem.A.shape[0], 10)
        assert_equal(problem.A.shape[1], 10)
        assert_equal(problem.B.shape[0], 10)
        assert_equal(problem.B.shape[1], 2)
        assert_equal(problem.C.shape[0], 2)
        assert_equal(problem.C.shape[1], 2)

        problem = skp.ProcrustesProblem((50,50,5,5), problemnumber=3)
        assert_equal(problem.A.shape[0], 50)
        assert_equal(problem.A.shape[1], 50)
        assert_equal(problem.B.shape[0], 50)
        assert_equal(problem.B.shape[1], 5)
        assert_equal(problem.C.shape[0], 5)
        assert_equal(problem.C.shape[1], 5)

    def test_setproblem_block_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10,10,3,3), problemnumber=1)

    def test_setproblem_dimensions_problem3(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10,10,5,5), problemnumber=3)

    def test_setproblem_singular_values_problem_1(self):
        problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)
        U, S, VT = sp.svd(problem.A)
        # Zhang & Du - Example 1
        #Sigmaorig = 10.0 + 2.0*np.random.rand(10)
        assert_array_less(S, 12*np.ones((10,)))
        assert_array_less(10*np.ones((10,)), S)

    def test_setproblem_singular_values_problem_2(self):
        problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=2)
        U, S, VT = sp.svd(problem.A)
        S = np.sort(S)
        # Zhang & Du - Example 3
        Sigmaorig = np.zeros(min(10,10))
        for i in range(0,min(10,10)):
            Sigmaorig[i] = 1.0 + (99.0*float(i-1))/(float(10)-1.0)
        Sigmaorig = np.sort(np.abs(Sigmaorig))
        assert_allclose(Sigmaorig, S, rtol=2.0)

    def test_setproblem_singular_values_problem_3(self):
        problem = skp.ProcrustesProblem((50,50,5,5), problemnumber=3)
        U, S, VT = sp.svd(problem.A)
        S = np.sort(S)
        # Zhang & Du - Example 4
        n1, n2, n3, n4 = (15, 15, 12, 8)
        vaux = np.zeros((50,))
        vaux[0:n1] = 10.0
        vaux[n1:n1+n2] = 5.0
        vaux[n1+n2:n1+n2+n3] = 2.0
        vaux[n1+n2+n3:] = 0
        Sigma = np.sort(vaux)
        assert_allclose(Sigma, S, rtol=1.0)

    def test_setproblem_known_solution_problem_1(self):
        problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)
        #Xsol = np.eye(n,p)
        #Xsol = np.random.permutation(Xsol)
        # Generate B
        #B = np.dot(A, np.dot(Xsol, C))
        assert_allclose(problem.B, np.dot(problem.A, \
                                          np.dot(problem.Xsol, problem.C)))

    def test_setproblem_known_solution_problem_2(self):
        problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=2)
        assert_allclose(problem.B, np.dot(problem.A, \
                                          np.dot(problem.Xsol, problem.C)))

    def test_setproblem_known_solution_problem_3(self):
        problem = skp.ProcrustesProblem((50,50,5,5), problemnumber=3)
        assert_allclose(problem.B, np.dot(problem.A, \
                                          np.dot(problem.Xsol, problem.C)))

    def test_setproblem_given_matrices_has_3_elements(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10,10,2,2), \
                      matrices=[np.random.rand(10,10), np.random.rand(10,2)])

    def test_setproblem_given_A_has_correct_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10,10,2,2), \
                      matrices=[np.random.rand(10,1), np.random.rand(10,2),
                                np.random.rand(2,2)])

    def test_setproblem_given_B_has_correct_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10,10,2,2), \
                      matrices=[np.random.rand(10,10), np.random.rand(1,2),
                                np.random.rand(2,2)])

    def test_setproblem_given_C_has_correct_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10,10,2,2), \
                      matrices=[np.random.rand(10,10), np.random.rand(10,2),
                                np.random.rand(10,2)])

    def test_setproblem_given_X_has_correct_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10,10,2,2), \
                      matrices=[np.random.rand(10,10), np.random.rand(10,2),
                                np.random.rand(2,2), np.random.rand(10,10)])

# Testing functions inside SPGSolver class:
class TestSPGSolver(TestCase):

    def setUp(self):
        self.problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)

    def test_setoptions(self):
        mysolver = skp.SPGSolver()
        assert_(mysolver.options is not None)

    def test_setoptions_full_results(self):
        assert_raises(Exception, skp.SPGSolver, full_results=" ")

    def test_setoptions_strategy(self):
        assert_raises(Exception, skp.SPGSolver, strategy=" ")

    def test_setoptions_gtol(self):
        assert_raises(Exception, skp.SPGSolver, gtol=" ")

    def test_setoptions_maxiter(self):
        assert_raises(Exception, skp.SPGSolver, maxiter=10.5)

    def test_setoptions_verbose(self):
        assert_raises(Exception, skp.SPGSolver, verbose=5)

    def test_setoptions_changevar(self):
        assert_raises(Exception, skp.SPGSolver, changevar=1)

# Testing functions inside GKBSolver class:
class TestGKBSolver(TestCase):

    def setUp(self):
        self.problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)

    def test_setoptions(self):
        mysolver = skp.GKBSolver()
        assert_(mysolver.options is not None)

    def test_setoptions_full_results(self):
        assert_raises(Exception, skp.GKBSolver, full_results=" ")

    def test_setoptions_strategy(self):
        assert_raises(Exception, skp.GKBSolver, strategy=" ")

    def test_setoptions_gtol(self):
        assert_raises(Exception, skp.GKBSolver, gtol=" ")

    def test_setoptions_maxiter(self):
        assert_raises(Exception, skp.GKBSolver, maxiter=10.5)

    def test_setoptions_verbose(self):
        assert_raises(Exception, skp.GKBSolver, verbose=5)

    def test_setoptions_changevar(self):
        assert_raises(Exception, skp.GKBSolver, changevar=1)

# Testing functions inside EBSolver class:
class TestEBSolver(TestCase):

    def setUp(self):
        self.problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)

    def test_setoptions(self):
        mysolver = skp.EBSolver()
        assert_(mysolver.options is not None)

    def test_setoptions_full_results(self):
        assert_raises(Exception, skp.EBSolver, full_results=" ")

    def test_setoptions_tol(self):
        assert_raises(Exception, skp.EBSolver, tol=" ")

    def test_setoptions_maxiter(self):
        assert_raises(Exception, skp.EBSolver, maxiter=10.5)

    def test_setoptions_verbose(self):
        assert_raises(Exception, skp.EBSolver, verbose=2)

# Testing functions inside GPISolver class:
class TestGPISolver(TestCase):

    def setUp(self):
        self.problem = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)

    def test_setoptions(self):
        mysolver = skp.GPISolver()
        assert_(mysolver.options is not None)

    def test_setoptions_full_results(self):
        assert_raises(Exception, skp.GPISolver, full_results=" ")

    def test_setoptions_tol(self):
        assert_raises(Exception, skp.GPISolver, tol=" ")

    def test_setoptions_maxiter(self):
        assert_raises(Exception, skp.GPISolver, maxiter=10.5)

    def test_setoptions_verbose(self):
        assert_raises(Exception, skp.GPISolver, verbose=2)

# Other functions
class TestSpectralSolver(TestCase):

    # spg solver
    # def test_spectral_solver_known_solution_spg_small(self):
    #     A = np.eye(10,10)
    #     C = np.eye(2,2)
    #     B = np.ones((10,2))
    #     problem = skp.ProcrustesProblem((10,10,2,2), matrices=(A, B, C))
    #     mysolver = skp.SPGSolver()
    #     result = mysolver.solve(problem)
    #     assert_allclose(result.fun, 13.055728090000844)
    #     assert_allclose(result.normgrad, 2.0330196556284077e-14)
    #     assert_equal(result.nbiter, 1)
    #     assert_equal(result.nfev, 2)

    def test_spectral_solver_known_solution_spg_problem_1(self):
        problem = skp.ProcrustesProblem((100,100,10,10), problemnumber=1)
        mysolver = skp.SPGSolver()
        result = mysolver.solve(problem)
        assert_array_less(result.nbiter, 8)

    def test_spectral_solver_known_solution_spg_problem_2(self):
        A = np.asarray([[-4.955319052527563883e+00, 9.490698890235075069e+00, \
                         3.871013427009882690e+01, -2.407812454150707637e+01],
                        [1.459652616043183571e+01, 5.183996942308650269e+00,  \
                         1.750998867584864627e+01, -3.465157159361613992e+01],
                        [-3.176193070766883864e+01, -1.488163256103923615e+01,\
                         -1.834308690911770512e+01, -3.493231059965589580e+00],
                        [1.386947318465110079e+01, -3.622322069989070137e+00, \
                         2.796986231299064318e+00, -3.324889768560630898e+01]])
        B = np.asarray([[3.871013427009882690e+01, 9.490698890235075069e+00],
                        [1.750998867584864627e+01, 5.183996942308650269e+00],
                        [-1.834308690911770512e+01, -1.488163256103923615e+01],
                        [2.796986231299064318e+00, -3.622322069989070137e+00]])
        C = np.eye(2,2)
        problem = skp.ProcrustesProblem((4,4,2,2), matrices=(A,B,C))
        mysolver = skp.SPGSolver()
        result = mysolver.solve(problem)
        assert_equal(result.nbiter, 26)
        assert_equal(result.nfev, 31)
        assert_allclose(result.fun, 3.589921974053535e-12)
        assert_allclose(result.normgrad, 0.00018803967459963403)

    # gkb solver
    # def test_spectral_solver_known_solution_gkb_small(self):
    #     A = np.eye(10,10)
    #     C = np.eye(2,2)
    #     B = np.ones((10,2))
    #     problem = skp.ProcrustesProblem((10,10,2,2), matrices=(A, B, C))
    #     mysolver = skp.GKBSolver()
    #     result = mysolver.solve(problem)
    #     assert_allclose(result.fun, 13.055728090000844) # local minimizer
    #     assert_allclose(result.normgrad, 3.8845908793929472e-15)
    #     assert_equal(result.blocksteps, 1)
    #     assert_allclose(result.nbiter, 0.8)
    #     assert_allclose(result.nfev, 1.2)

    def test_spectral_solver_known_solution_gkb_problem_1(self):
        problem = skp.ProcrustesProblem((100,100,10,10), problemnumber=1)
        mysolver = skp.GKBSolver()
        result = mysolver.solve(problem)
        assert_array_less(result.blocksteps, 7)

    def test_spectral_solver_known_solution_gkb_problem_2(self):
        A = np.asarray([[-4.955319052527563883e+00, 9.490698890235075069e+00, \
                         3.871013427009882690e+01, -2.407812454150707637e+01],
                        [1.459652616043183571e+01, 5.183996942308650269e+00,  \
                         1.750998867584864627e+01, -3.465157159361613992e+01],
                        [-3.176193070766883864e+01, -1.488163256103923615e+01,\
                         -1.834308690911770512e+01, -3.493231059965589580e+00],
                        [1.386947318465110079e+01, -3.622322069989070137e+00, \
                         2.796986231299064318e+00, -3.324889768560630898e+01]])
        B = np.asarray([[3.871013427009882690e+01, 9.490698890235075069e+00],
                        [1.750998867584864627e+01, 5.183996942308650269e+00],
                        [-1.834308690911770512e+01, -1.488163256103923615e+01],
                        [2.796986231299064318e+00, -3.622322069989070137e+00]])
        C = np.eye(2,2)
        problem = skp.ProcrustesProblem((4,4,2,2), matrices=(A,B,C))
        mysolver = skp.GKBSolver()
        result = mysolver.solve(problem)
        assert_allclose(result.nbiter, 28)
        assert_allclose(result.nfev, 46)
        assert_allclose(result.fun, 1.940760937201845e-13)
        assert_allclose(result.normgrad, 0.00010351602115043701)

    #TODO: test known solution for problem 3.

class TestBlockBidiag(TestCase):

    # def test_blockbidiag(self):

    #     # m = 3
    #     # n = 3
    #     # q = 3
    #     # nsteps = 1
    #     # partial = 0
    #     # A = np.array([[0,2,1],[1,1,2],[0,0,3]])
    #     # B = np.copy(A)
    #     # Qtrue = np.array([[0,1,0],[1,0,0],[0,0,1]])
    #     # Rtrue = np.array([[1,1,2],[0,2,1],[0,0,3]])

    #     m,n,p,q = (6,6,2,2)
    #     nsteps = 0
    #     partial = 0
    #     Qtrue = np.array([[1,0,0,0,0,0],
    #                       [0,0,1,0,0,0],
    #                       [0,0,0,1,0,0],
    #                       [0,1,0,0,0,0],
    #                       [0,0,0,0,0,1],
    #                       [0,0,0,0,1,0]])
    #     Rtrue = np.array([[1,2,3,4,5,6],
    #                       [0,1,2,3,4,5],
    #                       [0,0,1,2,3,4],
    #                       [0,0,0,1,2,3],
    #                       [0,0,0,0,1,2],
    #                       [0,0,0,0,0,1]])
    #     A = np.dot(Qtrue,Rtrue)
    #     B = np.dot(A, np.ones((n,p)))
    #     C = np.eye(p,q)

    #     U = np.zeros((m,m))
    #     V = np.zeros((n,n))
    #     T = np.zeros((m,n+q))

    #     problem = skp.ProcrustesProblem((m,n,p,q), matrices=(A,B,C))

    #     U, V, T, B1, reorth = skp.blockbidiag(problem, U, V, T, nsteps, partial)

        # print("\nT = {}\n".format(T[0:largedim, 0:smalldim]))
        # print("U = {}\n".format(U[0:m, 0:largedim]))
        # print("V = {}\n".format(V[0:n, 0:smalldim]))
        # print("T - UT*A*V = {}\n".format(T[0:largedim, 0:smalldim] - np.dot(U[0:m, 0:largedim].T, np.dot(A, V[0:n, 0:smalldim]))))

        maxerror = np.max(T[:,0:n] - np.dot(U.T, np.dot(A, V)))
        assert_allclose(maxerror, 3.081431772960419e-15)

class TestBidiagGs(TestCase):

    def test_bidiaggs(self):

        A = np.array([[0,0,3],[1,3,4],[0,2,1]])
        Q2 = np.eye(3,3)
        Q2, R2, reorth = skp.bidiaggs(0, A, Q2, 1e-10, 0)
        erro_bidiag = sp.norm(np.dot(Q2,R2) - A)
        assert_allclose(erro_bidiag, 0)

class TestEBSolver(TestCase):

    # def test_eb_solver_known_solution_small(self):
    #     A = np.eye(10,10)
    #     C = np.eye(2,2)
    #     B = np.ones((10,2))
    #     problem = skp.ProcrustesProblem((10,10,2,2), matrices=(A, B, C))
    #     mysolver = skp.EBSolver()
    #     result = mysolver.solve(problem)
    #     assert_allclose(result.fun, 3.6132711066291225)
    #     assert_equal(result.nbiter, 1)
    #     assert_equal(result.nfev, 2)

     def test_eb_solver_known_solution_problem_1(self):
         A = np.random.rand(4,4)
         C = np.eye(2,2)
         Xsol = np.eye(4,2)
         B = np.dot(A, np.dot(C, X))
         problem = skp.ProcrustesProblem((4,4,2,2), matrices=(A,B,C))
         mysolver = skp.EBSolver()
         result = mysolver.solve(problem)
         assert_allclose(result.solution, np.eye(4,2), 1e-1)
         #assert_allclose(result.nfev, 21)
         #assert_allclose(result.fun, 6.021465862884726e-07)

    # def test_eb_solver_known_solution_problem_2(self):
    #     A = np.asarray([[7.256898668268666697e+00, -2.573732000807788012e+01,\
    #                      6.009589673994661929e+00, -2.416566316285056715e+01],
    #                     [-1.050865070883216568e+01, 8.728310568335615471e+00,\
    #                      -2.396564993299465485e+00, 6.567064275964866304e+00],\
    #                     [-2.563405505463448009e+01, -1.522485768408047591e+01,\
    #                      1.057260685036533232e+01, 9.924230068539865002e+00],\
    #                     [6.125480264795922558e+00, -1.513009768260202392e+01,\
    #                      -5.846843858482707823e+00, 6.273222470266802731e+01]])
    #     B = np.asarray([[7.256898668268666697e+00, 6.009589673994661929e+00],
    #                     [-1.050865070883216568e+01, -2.396564993299465485e+00],
    #                     [-2.563405505463448009e+01, 1.057260685036533232e+01],
    #                     [6.125480264795922558e+00, -5.846843858482707823e+00]])
    #     C = np.eye(2,2)
    #     problem = skp.ProcrustesProblem((4,4,2,2), matrices=(A,B,C))
    #     mysolver = skp.EBSolver()
    #     result = mysolver.solve(problem)
    #     assert_allclose(result.nfev, 24)
    #     assert_allclose(result.fun, 5.881556898206642e-07)

    # TODO add known solution test for problem 3
    # TODO in the future, if we allow non square problems
    # def test_set_options_eb_solver_A_not_square(self):
    #     problem = skp.ProcrustesProblem((4,2,2,2), problemnumber=1)
    #     assert_raises(Exception, problem.set_options, solver="eb")

    # def test_set_options_eb_solver_C_not_square(self):
    #     problem = skp.ProcrustesProblem((4,4,4,2), problemnumber=1)
    #     assert_raises(Exception, problem.set_options, solver="eb")

    # def test_set_options_eb_solver_C_must_be_eye(self):
    #     problem = skp.ProcrustesProblem((4,2,2,2), problemnumber=1)
    #     problem.C = np.random.rand(2,2)
    #     assert_raises(Exception, problem.set_options, solver="eb")


class TestGPISolver(TestCase):

    def test_gpi_solver_known_solution_small(self):
        A = np.eye(10,10)
        C = np.eye(2,2)
        B = np.ones((10,2))
        problem = skp.ProcrustesProblem((10,10,2,2), matrices=(A, B, C))
        mysolver = skp.GPISolver()
        result = mysolver.solve(problem)
        assert_allclose(result.fun, 13.055728090000844)
        assert_equal(result.nbiter, 2)
        assert_equal(result.nfev, 3)

    # def test_gpi_solver_known_solution_problem_1(self):
    #     A = np.asarray([[5.520962350152090359e+00, -6.793095607530246660e+00, \
    #                    -2.968126930397382957e+00, 5.215698321012116168e+00],
    #                   [-6.775745407471069015e+00, 1.822680160805416616e+00,\
    #                    -4.224183825647802593e+00, 7.178797488759216527e+00],
    #                   [6.355695000758535329e+00, 8.514938821664049584e+00,\
    #                    4.926367379292874715e-01, 3.889893914905437455e+00],
    #                   [-6.186446461568055888e-01, -8.111306483700752024e-01,\
    #                    9.932080492258702265e+00, 4.599731376522814053e+00]])
    #     B = np.asarray([[-6.793095607530246660e+00, 5.215698321012116168e+00],
    #                     [1.822680160805416616e+00, 7.178797488759216527e+00],
    #                     [8.514938821664049584e+00, 3.889893914905437455e+00],
    #                     [-8.111306483700752024e-01, 4.599731376522814053e+00]])

    #     C = np.eye(2,2)
    #     problem = skp.ProcrustesProblem((4,4,2,2), matrices=(A,B,C))
    #     mysolver = skp.GPISolver()
    #     result = mysolver.solve(problem)
    #     assert_allclose(result.nfev, 7)
    #     assert_allclose(result.fun, 5.868725008379679e-07)

    # def test_gpi_solver_known_solution_problem_2(self):
    #     A = np.asarray([[7.256898668268666697e+00, -2.573732000807788012e+01,\
    #                      6.009589673994661929e+00, -2.416566316285056715e+01],
    #                     [-1.050865070883216568e+01, 8.728310568335615471e+00,\
    #                      -2.396564993299465485e+00, 6.567064275964866304e+00],\
    #                     [-2.563405505463448009e+01, -1.522485768408047591e+01,\
    #                      1.057260685036533232e+01, 9.924230068539865002e+00],\
    #                     [6.125480264795922558e+00, -1.513009768260202392e+01,\
    #                      -5.846843858482707823e+00, 6.273222470266802731e+01]])
    #     B = np.asarray([[7.256898668268666697e+00, 6.009589673994661929e+00],
    #                     [-1.050865070883216568e+01, -2.396564993299465485e+00],
    #                     [-2.563405505463448009e+01, 1.057260685036533232e+01],
    #                     [6.125480264795922558e+00, -5.846843858482707823e+00]])
    #     C = np.eye(2,2)
    #     problem = skp.ProcrustesProblem((4,4,2,2), matrices=(A,B,C))
    #     mysolver = skp.GPISolver()
    #     result = mysolver.solve(problem)
    #     assert_allclose(result.nfev, 80)
    #     assert_allclose(result.fun, 4.320064300629071e-06)

    # TODO add known solution test for problem 3

if __name__ == "__main__":
    run_module_suite()
