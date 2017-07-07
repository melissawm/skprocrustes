=================
scikit-procrustes
=================

Collection of solvers for the (Weighted) Orthogonal Procrustes Problem.

   .. math::

      \min  {\| AXC-B \|}_F^2 \qquad s. t. \quad X^TX=I

where :math:`A_{m \times n}, B_{m \times q}, C_{p \times q}, 
X_{n \times p}`. Usually n >> p, which means we can solve unbalanced problems.

Available Solvers
=================

* :ref:`spg <spg>`    Nonmonotone Spectral Projected Gradient Method for the (unbalanced) WOPP, as described in [1]_.
* :ref:`gkb <gkb>`    Nonmonotone Spectral Projected Gradient Method using incomplete Lanczos (Golub-Kahan) Bidiagonalization, as described in [2]_.
* :ref:`eb <eb>`      Expansion-Balance method, as described in [3]_.
* :ref:`gpi <gpi>`    Generalized Power Iteration for the WOPP, as described in [5]_.
     
Usage
=====

To use the package to solve a given problem with predefined matrices A, B and C
using the SPG solver, for example, use

    >>> import skprocrustes as skp
    >>> problem = skp.ProcrustesProblem((m,n,p,q),     # tuple
                                        matrices=[A, B, C])
    >>> mysolver = skp.SPGSolver(**kwargs)
    >>> result = mysolver.solve(problem)

where `**kwargs` are the selected solver's options (see the `Module Reference <skprocrustes.html>`_ for more details).

To use the package to solve one of the three predefined problems (as described in [4]_), using the GKB solver, for example, use

    >>> import skprocrustes as skp
    >>> problem = skp.ProcrustesProblem((m,n,p,q),     # tuple
                                        problemnumber=1)
    >>> mysolver = skp.GKBSolver(**kwargs)
    >>> result = mysolver.solve(problem)

    
.. _bibliography:

References
==========

.. [1] J.B. Francisco, F.S. Viloche Bazán, Nonmonotone algorithm for minimization on closed sets with applications to minimization on Stiefel manifolds, Journal of Computational and Applied Mathematics, 2012, 236(10): 2717--2727 <http://dx.doi.org/10.1016/j.cam.2012.01.014>
.. [2] J.B. Francisco, F.S. Viloche Bazán and M. Weber Mendonça, Non-monotone algorithm for minimization on arbitrary domains with applications to large-scale orthogonal Procrustes problem, Applied Numerical Mathematics, 2017, 112: 51--64 <https://doi.org/10.1016/j.apnum.2016.09.018>
.. [3] J.M.F. ten Berge and D.L. Knol, Orthogonal rotations to maximal agreement for two or more matrices of different column orders, Psychometrika 1984, 49: 49--55 <https://doi:10.1007/BF02294205>
.. [4] Z. Zhang, K. Du, Successive projection method for solving the unbalanced Procrustes problem, Sci. China Ser. A, 2006, 49: 971–986.
.. [5] F. Nie, R. Zhang, X. Li, A generalized power iteration method for solving quadratic problem on the Stiefel manifold, Sci. China Inf. Sci., 2017, 60: 112101:1--112101:10. <http://dx.doi.org/10.1007/s11432-016-9021-9>


Installation
============

Quick Installation
------------------

In the root directory of the package, just do::

   python setup.py install

Latest Software
---------------
The latest software can be downloaded from `GitHub <https://github.com/melissawm/scikit-procrustes>`_

Installation Dependencies
-------------------------
``scikit-procrustes`` requires the following software packages to be
installed:

* `Python <http://www.python.org>`_ 3.6.1 or later.
* `NumPy <http://www.numpy.org>`_ 1.13.0 or later.
* `SciPy <http://www.scipy.org>`_ 0.19.0 or later.
* `Matplotlib <http://www.matplotlib.org>`_ 2.0.2 or later.

Contents
========

.. toctree::
   :maxdepth: 2
	      
   Module Reference <skprocrustes>
   Solvers <solvers>
   License <license>
   Authors <authors>
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

