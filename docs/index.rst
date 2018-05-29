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

* ``SPGSolver``    Nonmonotone Spectral Projected Gradient Method for the (unbalanced) WOPP, as described in :cite:`FranBaza12`.
* ``GKBSolver``    Nonmonotone Spectral Projected Gradient Method using incomplete Lanczos (Golub-Kahan) Bidiagonalization, as described in :cite:`FranBazaWebe17`.
* ``EBSolver``     Expansion-Balance method, as described in :cite:`BergKnol84`.
* ``GPISolver``    Generalized Power Iteration for the WOPP, as described in :cite:`NieZhanLi17`.
     
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

To use the package to solve one of the three predefined problems (as described in :cite:`ZhanDu06`), using the GKB solver, for example, use

    >>> import skprocrustes as skp
    >>> problem = skp.ProcrustesProblem((m,n,p,q),     # tuple
                                        problemnumber=1)
    >>> mysolver = skp.GKBSolver(**kwargs)
    >>> result = mysolver.solve(problem)

References
==========
    
.. bibliography:: references.bib

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
   License <license>
   Authors <authors>
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

