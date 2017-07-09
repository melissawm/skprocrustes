#!/usr/bin/env python

import os
import sys
import subprocess

from setuptools import find_packages, setup


# Dependencies.
with open('requirements.txt') as f:
    tests_require = f.readlines()
install_requires = [t.strip() for t in tests_require]

descr = """\
Solvers for the Weighted Orthogonal Procrustes Problem (WOPP)

    min norm(AXC-B,'fro')**2
    s/t X'*X=I

where A(m,n), B(m,q), C(p,q) and X(n,p). (usually n >> p, which
means we can solve unbalanced problems.)
"""

# Documentation building command
try:
    from sphinx.setup_command import BuildDoc as SphinxBuildDoc

    class BuildDoc(SphinxBuildDoc):
        """Run in-place build before Sphinx doc build"""
        def run(self):
            ret = subprocess.call(
                [sys.executable, sys.argv[0], 'build_ext', '-i']
            )
            if ret != 0:
                raise RuntimeError("Building Scipy failed!")
            SphinxBuildDoc.run(self)
    cmdclass = {'build_sphinx': BuildDoc}
except ImportError:
    cmdclass = {}


def get_version():
    """Obtain the version number"""
    import imp
    mod = imp.load_source(
        'version', os.path.join('skprocrustes', 'version.py')
    )
    return mod.__version__


config = dict(
    name='scikit-procrustes',
    version=get_version(),
    description='Package of solvers for the Orthogonal Procrustes Problem',
    long_description=descr,
    author='Melissa Weber Mendonça',
    author_email='melissawm@gmail.com',
    url='https://github.com/melissawm/skprocrustes',
    license='BSD',
    download_url='https://pypi.org/project/scikit-procrustes',
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Topic :: Scientific/Engineering :: Mathematics'
    ],
    packages=find_packages(),
    include_package_data=True,
    cmdclass=cmdclass,
    install_requires=install_requires,
)


setup(**config)
