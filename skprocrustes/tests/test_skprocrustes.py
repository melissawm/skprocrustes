import numpy as np
from numpy.testing import (assert_raises, assert_allclose, assert_equal,
                           assert_, TestCase, run_module_suite,
                           assert_array_less)
from scipy import linalg as sp
import skprocrustes as skp


# Testing functions inside ProcrustesProblem class:
class TestSetProblem(TestCase):

    # def _setproblem(self, matrices, problemnumber):
    # A(m,n), B(m,q), C(p,q) -> problem matrices

    def test_setproblem_dimensions_square(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10, 2, 5, 10),
                      problemnumber=1)

    def test_setproblem_dimensions(self):
        problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
        assert_equal(problem.A.shape[0], 10)
        assert_equal(problem.A.shape[1], 10)
        assert_equal(problem.B.shape[0], 10)
        assert_equal(problem.B.shape[1], 2)
        assert_equal(problem.C.shape[0], 2)
        assert_equal(problem.C.shape[1], 2)

        problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
        assert_equal(problem.A.shape[0], 10)
        assert_equal(problem.A.shape[1], 10)
        assert_equal(problem.B.shape[0], 10)
        assert_equal(problem.B.shape[1], 2)
        assert_equal(problem.C.shape[0], 2)
        assert_equal(problem.C.shape[1], 2)

        problem = skp.ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
        assert_equal(problem.A.shape[0], 50)
        assert_equal(problem.A.shape[1], 50)
        assert_equal(problem.B.shape[0], 50)
        assert_equal(problem.B.shape[1], 5)
        assert_equal(problem.C.shape[0], 5)
        assert_equal(problem.C.shape[1], 5)

    def test_setproblem_block_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10, 10, 3, 3),
                      problemnumber=1)

    def test_setproblem_dimensions_problem3(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10, 10, 5, 5),
                      problemnumber=3)

    def test_setproblem_singular_values_problem_1(self):
        problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
        U, S, VT = sp.svd(problem.A)
        # Zhang & Du - Example 1
        # Sigmaorig = 10.0 + 2.0*np.random.rand(10)
        assert_array_less(S, 12*np.ones((10,)))
        assert_array_less(10*np.ones((10,)), S)

    def test_setproblem_singular_values_problem_2(self):
        problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
        U, S, VT = sp.svd(problem.A)
        S = np.sort(S)
        # Zhang & Du - Example 3
        Sigmaorig = np.zeros(min(10, 10))
        for i in range(0, min(10, 10)):
            Sigmaorig[i] = 1.0 + (99.0*float(i-1))/(float(10)-1.0)
        Sigmaorig = np.sort(np.abs(Sigmaorig))
        assert_allclose(Sigmaorig, S, rtol=2.0)

    def test_setproblem_singular_values_problem_3(self):
        problem = skp.ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
        U, S, VT = sp.svd(problem.A)
        S = np.sort(S)
        # Zhang & Du - Example 4
        n1, n2, n3 = (15, 15, 12)
        vaux = np.zeros((50,))
        vaux[0:n1] = 10.0
        vaux[n1:n1+n2] = 5.0
        vaux[n1+n2:n1+n2+n3] = 2.0
        vaux[n1+n2+n3:] = 0
        Sigma = np.sort(vaux)
        assert_allclose(Sigma, S, rtol=1.0)

    def test_setproblem_known_solution_problem_1(self):
        problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
        # Xsol = np.eye(n,p)
        # Xsol = np.random.permutation(Xsol)
        # Generate B
        # B = np.dot(A, np.dot(Xsol, C))
        assert_allclose(problem.B, np.dot(problem.A,
                                          np.dot(problem.Xsol, problem.C)))

    def test_setproblem_known_solution_problem_2(self):
        problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
        assert_allclose(problem.B, np.dot(problem.A,
                                          np.dot(problem.Xsol, problem.C)))

    def test_setproblem_known_solution_problem_3(self):
        problem = skp.ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
        assert_allclose(problem.B, np.dot(problem.A,
                                          np.dot(problem.Xsol, problem.C)))

    def test_setproblem_given_matrices_has_3_elements(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10, 10, 2, 2),
                      matrices=[np.random.rand(10, 10), np.random.rand(10, 2)])

    def test_setproblem_given_A_has_correct_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10, 10, 2, 2),
                      matrices=[np.random.rand(10, 1), np.random.rand(10, 2),
                                np.random.rand(2, 2)])

    def test_setproblem_given_B_has_correct_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10, 10, 2, 2),
                      matrices=[np.random.rand(10, 10), np.random.rand(1, 2),
                                np.random.rand(2, 2)])

    def test_setproblem_given_C_has_correct_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10, 10, 2, 2),
                      matrices=[np.random.rand(10, 10), np.random.rand(10, 2),
                                np.random.rand(10, 2)])

    def test_setproblem_given_X_has_correct_dimensions(self):
        assert_raises(Exception, skp.ProcrustesProblem, (10, 10, 2, 2),
                      matrices=[np.random.rand(10, 10), np.random.rand(10, 2),
                                np.random.rand(2, 2), np.random.rand(10, 10)])


# Testing functions inside SPGSolver class:
class TestSPGSolver(TestCase):

    def setUp(self):
        self.problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)

    def test_setoptions(self):
        mysolver = skp.SPGSolver()
        assert_(mysolver.options is not None)

    def test_setoptions_full_results(self):
        assert_raises(Exception, skp.SPGSolver, full_results=" ")

    def test_setoptions_filename(self):
        assert_raises(Exception, skp.SPGSolver, filename=1)

    def test_setoptions_strategy(self):
        assert_raises(Exception, skp.SPGSolver, strategy=" ")

    def test_setoptions_gtol(self):
        assert_raises(Exception, skp.SPGSolver, gtol=" ")

    def test_setoptions_eta(self):
        assert_raises(Exception, skp.SPGSolver, eta=" ")

    def test_setoptions_etavar(self):
        assert_raises(Exception, skp.SPGSolver, etavar=1)

    def test_setoptions_maxiter(self):
        assert_raises(Exception, skp.SPGSolver, maxiter=10.5)

    def test_setoptions_verbose(self):
        assert_raises(Exception, skp.SPGSolver, verbose=5)

    def test_setoptions_changevar(self):
        assert_raises(Exception, skp.SPGSolver, changevar=1)

    def test_setoptions_bloboptest(self):
        assert_raises(Exception, skp.SPGSolver, bloboptest=1)

    def test_setoptions_polar(self):
        assert_raises(Exception, skp.SPGSolver, polar=1)

    def test_setoptions_timer(self):
        assert_raises(Exception, skp.SPGSolver, timer=3)

    def test_setoptions_precond(self):
        assert_raises(Exception, skp.SPGSolver, precond=1)

        
# Testing functions inside GKBSolver class:
class TestGKBSolver(TestCase):

    def setUp(self):
        self.problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)

    def test_setoptions(self):
        mysolver = skp.GKBSolver()
        assert_(mysolver.options is not None)

    def test_setoptions_full_results(self):
        assert_raises(Exception, skp.GKBSolver, full_results=" ")

    def test_setoptions_filename(self):
        assert_raises(Exception, skp.GKBSolver, filename=1)

    def test_setoptions_strategy(self):
        assert_raises(Exception, skp.GKBSolver, strategy=" ")

    def test_setoptions_gtol(self):
        assert_raises(Exception, skp.GKBSolver, gtol=" ")

    def test_setoptions_eta(self):
        assert_raises(Exception, skp.GKBSolver, eta=" ")

    def test_setoptions_etavar(self):
        assert_raises(Exception, skp.GKBSolver, etavar=1)

    def test_setoptions_maxiter(self):
        assert_raises(Exception, skp.GKBSolver, maxiter=10.5)

    def test_setoptions_verbose(self):
        assert_raises(Exception, skp.GKBSolver, verbose=5)

    def test_setoptions_changevar(self):
        assert_raises(Exception, skp.GKBSolver, changevar=1)

    def test_setoptions_bloboptest(self):
        assert_raises(Exception, skp.GKBSolver, bloboptest=1)

    def test_setoptions_polar(self):
        assert_raises(Exception, skp.GKBSolver, polar=1)

    def test_setoptions_timer(self):
        assert_raises(Exception, skp.GKBSolver, timer=3)

    def test_setoptions_precond(self):
        assert_raises(Exception, skp.SPGSolver, precond=1)


# Testing functions inside EBSolver class:
class TestEBSolver(TestCase):

    def setUp(self):
        self.problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)

    def test_setoptions(self):
        mysolver = skp.EBSolver()
        assert_(mysolver.options is not None)

    def test_setoptions_full_results(self):
        assert_raises(Exception, skp.EBSolver, full_results=" ")

    def test_setoptions_filename(self):
        assert_raises(Exception, skp.EBSolver, filename=1)

    def test_setoptions_tol(self):
        assert_raises(Exception, skp.EBSolver, tol=" ")

    def test_setoptions_maxiter(self):
        assert_raises(Exception, skp.EBSolver, maxiter=10.5)

    def test_setoptions_verbose(self):
        assert_raises(Exception, skp.EBSolver, verbose=2)

    def test_setoptions_timer(self):
        assert_raises(Exception, skp.EBSolver, timer=3)


# Testing functions inside GPISolver class:
class TestGPISolver(TestCase):

    def setUp(self):
        self.problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)

    def test_setoptions(self):
        mysolver = skp.GPISolver()
        assert_(mysolver.options is not None)

    def test_setoptions_full_results(self):
        assert_raises(Exception, skp.GPISolver, full_results=" ")

    def test_setoptions_filename(self):
        assert_raises(Exception, skp.GPISolver, filename=1)

    def test_setoptions_tol(self):
        assert_raises(Exception, skp.GPISolver, tol=" ")

    def test_setoptions_maxiter(self):
        assert_raises(Exception, skp.GPISolver, maxiter=10.5)

    def test_setoptions_verbose(self):
        assert_raises(Exception, skp.GPISolver, verbose=2)

    def test_setoptions_timer(self):
        assert_raises(Exception, skp.GPISolver, timer=3)


# Other functions
class TestSpectralSolver(TestCase):

    # spg solver
    def test_spectral_solver_known_solution_spg_small(self):
        A = np.eye(10, 10)
        C = np.eye(2, 2)
        Xsol = np.eye(10, 2)
        B = np.dot(A, np.dot(Xsol, C))
        problem = skp.ProcrustesProblem((10, 10, 2, 2), matrices=(A, B, C))
        mysolver = skp.SPGSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, np.eye(10, 2), atol=1e-3)

    def test_spectral_solver_known_solution_spg_problem_1(self):
        problem = skp.ProcrustesProblem((100, 100, 10, 10), problemnumber=1)
        mysolver = skp.SPGSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_known_solution_spg_problem_2(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=2)
        mysolver = skp.SPGSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_spg_eta(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.SPGSolver(verbose=0, eta=0.1)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_spg_etavar(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.SPGSolver(verbose=0, etavar=True)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_spg_filename(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.SPGSolver(verbose=0, filename="testspgfilename.txt")
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    # TODO test below is failing. why?
    ### def test_spectral_solver_spg_changevar(self):
    ###     problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    ###     mysolver = skp.SPGSolver(verbose=0, changevar=True)
    ###     result = mysolver.solve(problem)
    ###     assert_allclose(result.solution, problem.Xsol, atol=1e-3)

#    def test_spectral_solver_spg_constraint_violation(self):
#    não sei como fazer

    # gkb solver
    def test_spectral_solver_known_solution_gkb_small(self):
        A = np.eye(10, 10)
        C = np.eye(2, 2)
        Xsol = np.eye(10, 2)
        B = np.dot(A, np.dot(Xsol, C))
        problem = skp.ProcrustesProblem((10, 10, 2, 2), matrices=(A, B, C))
        mysolver = skp.GKBSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, np.eye(10, 2), atol=1e-3)

    def test_spectral_solver_known_solution_gkb_problem_1(self):
        problem = skp.ProcrustesProblem((100, 100, 10, 10), problemnumber=1)
        mysolver = skp.GKBSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_known_solution_gkb_problem_2(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=2)
        mysolver = skp.GKBSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    # TODO: test known solution for problem 3.

    def test_spectral_solver_gkb_polar_None(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.GKBSolver(verbose=0, polar=None)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_gkb_polar_ns(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.GKBSolver(verbose=0, polar="ns")
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_gkb_eta(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.GKBSolver(verbose=0, eta=0.1)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_gkb_etavar(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.GKBSolver(verbose=0, etavar=True)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_gkb_filename(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.GKBSolver(verbose=0, filename="testspgfilename.txt")
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    def test_spectral_solver_gkb_changevar(self):
        # this test makes no sense?
        # TODO fix this
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.GKBSolver(verbose=0, changevar=True)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

class TestBlockBidiag(TestCase):

    def test_blockbidiag(self):

        # m = 3
        # n = 3
        # q = 3
        # nsteps = 1
        # partial = 0
        # A = np.array([[0,2,1],[1,1,2],[0,0,3]])
        # B = np.copy(A)
        # Qtrue = np.array([[0,1,0],[1,0,0],[0,0,1]])
        # Rtrue = np.array([[1,1,2],[0,2,1],[0,0,3]])

        m, n, p, q = (6, 6, 2, 2)
        nsteps = 0
        partial = 0
        Qtrue = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0]])
        Rtrue = np.array([[1, 2, 3, 4, 5, 6],
                          [0, 1, 2, 3, 4, 5],
                          [0, 0, 1, 2, 3, 4],
                          [0, 0, 0, 1, 2, 3],
                          [0, 0, 0, 0, 1, 2],
                          [0, 0, 0, 0, 0, 1]])
        A = np.dot(Qtrue, Rtrue)
        B = np.dot(A, np.ones((n, p)))
        C = np.eye(p, q)

        U = np.zeros((m, m))
        V = np.zeros((n, n))
        T = np.zeros((m, n+q))

        problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C))

        halfreorth = False
        U, V, T, B1, reorth = skp.blockbidiag(problem, U, V, T, nsteps,
                                              partial, halfreorth)

        # print("\nT = {}\n".format(T[0:largedim, 0:smalldim]))
        # print("U = {}\n".format(U[0:m, 0:largedim]))
        # print("V = {}\n".format(V[0:n, 0:smalldim]))
        # print("T - UT*A*V = {}\n".format(T[0:largedim, 0:smalldim] - \
        #    np.dot(U[0:m, 0:largedim].T, np.dot(A, V[0:n, 0:smalldim]))))

        maxerror = np.max(T[:, 0:n] - np.dot(U.T, np.dot(A, V)))
        assert_allclose(maxerror, 0, atol=1e-10)

    def test_blockbidiag_halfreorth(self):

        # m = 3
        # n = 3
        # q = 3
        # nsteps = 1
        # partial = 0
        # A = np.array([[0,2,1],[1,1,2],[0,0,3]])
        # B = np.copy(A)
        # Qtrue = np.array([[0,1,0],[1,0,0],[0,0,1]])
        # Rtrue = np.array([[1,1,2],[0,2,1],[0,0,3]])

        m, n, p, q = (6, 6, 2, 2)
        nsteps = 0
        partial = 0
        Qtrue = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0]])
        Rtrue = np.array([[1, 2, 3, 4, 5, 6],
                          [0, 1, 2, 3, 4, 5],
                          [0, 0, 1, 2, 3, 4],
                          [0, 0, 0, 1, 2, 3],
                          [0, 0, 0, 0, 1, 2],
                          [0, 0, 0, 0, 0, 1]])
        A = np.dot(Qtrue, Rtrue)
        B = np.dot(A, np.ones((n, p)))
        C = np.eye(p, q)

        U = np.zeros((m, m))
        V = np.zeros((n, n))
        T = np.zeros((m, n+q))

        problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C))

        halfreorth = True
        U, V, T, B1, reorth = skp.blockbidiag(problem, U, V, T, nsteps,
                                              partial, halfreorth)

        # print("\nT = {}\n".format(T[0:largedim, 0:smalldim]))
        # print("U = {}\n".format(U[0:m, 0:largedim]))
        # print("V = {}\n".format(V[0:n, 0:smalldim]))
        # print("T - UT*A*V = {}\n".format(T[0:largedim, 0:smalldim] - \
        #    np.dot(U[0:m, 0:largedim].T, np.dot(A, V[0:n, 0:smalldim]))))

        maxerror = np.max(T[:, 0:n] - np.dot(U.T, np.dot(A, V)))
        assert_allclose(maxerror, 0, atol=1e-10)


class TestBidiagGs(TestCase):

    def test_bidiaggs(self):

        A = np.array([[0, 0, 3], [1, 3, 4], [0, 2, 1]])
        Q2 = np.eye(3, 3)
        Q2, R2, reorth = skp.bidiaggs(0, A, Q2, 1e-10, 0)
        erro_bidiag = sp.norm(np.dot(Q2, R2) - A)
        assert_allclose(erro_bidiag, 0)

    # test bidiaggs with halfreorth


class TestEB_Solver(TestCase):

    def test_eb_solver_known_solution_small(self):
        A = np.eye(10, 10)
        C = np.eye(2, 2)
        Xsol = np.eye(10, 2)
        B = np.dot(A, np.dot(Xsol, C))
        problem = skp.ProcrustesProblem((10, 10, 2, 2), matrices=(A, B, C))
        mysolver = skp.EBSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.fun, 1e-6, atol=1e-2)
        assert_allclose(result.solution, np.eye(10, 2), atol=1e-2)

    def test_eb_solver_known_solution_problem_1(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.EBSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)
        assert_allclose(result.fun, 1e-6, atol=1e-2)

    def test_eb_solver_known_solution_problem_2(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=2)
        mysolver = skp.EBSolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)
        assert_allclose(result.fun, 1e-6, atol=1e-2)

    def test_eb_solver_filename(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.EBSolver(verbose=0, filename="testebfilename.txt")
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)

    # EBSolver does not work for problem 3
    # def test_eb_solver_known_solution_problem_3(self):
    #     problem = skp.ProcrustesProblem((50,50,5,5), problemnumber=3)
    #     mysolver = skp.EBSolver(verbose=0)
    #     result = mysolver.solve(problem)
    #     assert_allclose(result.solution, problem.Xsol, atol=1e-1)
    #     assert_allclose(result.fun, 1e-6, atol=1e-1)

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


class TestGPI_Solver(TestCase):

    def test_gpi_solver_known_solution_small(self):
        A = np.eye(10, 10)
        C = np.eye(2, 2)
        B = np.ones((10, 2))
        problem = skp.ProcrustesProblem((10, 10, 2, 2), matrices=(A, B, C))
        mysolver = skp.GPISolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.fun, 13.055728090000844)
        assert_equal(result.nbiter, 2)
        assert_equal(result.nfev, 3)

    def test_gpi_solver_known_solution_problem_1(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.GPISolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-3)
        assert_allclose(result.fun, 1e-6, atol=1e-2)

    def test_gpi_solver_known_solution_problem_2(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=2)
        mysolver = skp.GPISolver(verbose=0)
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-2)
        #assert_allclose(result.fun, 1e-6, atol=1e-2)

    def test_gpi_solver_filename(self):
        problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
        mysolver = skp.GPISolver(verbose=0, filename="testgpifilename.txt")
        result = mysolver.solve(problem)
        assert_allclose(result.solution, problem.Xsol, atol=1e-2)

    # GPISolver does not solve problem 3


class TestComputeResidual(TestCase):

    def test_compute_residual_no_precond(self):
        A = np.eye(10, 10)
        C = np.eye(2, 2)
        X = np.ones((10, 2))
        B = np.dot(A, np.dot(X, C))
        precond = None
        R, residual = skp.compute_residual(A, B, C, X, precond)
        assert_allclose(residual, sp.norm(np.dot(A, np.dot(X, C))-B, 'fro')**2)
        assert_allclose(R, np.dot(A, np.dot(X, C))-B)

    
if __name__ == "__main__":
    run_module_suite()
