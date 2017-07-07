#!/usr/bin/env python
descr = """\
Solvers for the Weighted Orthogonal Procrustes Problem (WOPP)

    min norm(AXC-B,'fro')**2
    s/t X'*X=I

where A(m,n), B(m,q), C(p,q) and X(n,p). (usually n >> p, which
means we can solve unbalanced problems.)
"""

DISTNAME            = 'scikit-procrustes'
DESCRIPTION         = 'Package of solvers for the Orthogonal Procrustes Problem'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Melissa Weber Mendonça',
MAINTAINER_EMAIL    = 'melissawm@gmail.com',
URL                 = 'https://github.com/melissawm/skprocrustes'
LICENSE             = 'BSD'
DOWNLOAD_URL        = URL
PACKAGE_NAME        = 'skprocrustes'
EXTRA_INFO          = dict(
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Topic :: Scientific/Engineering :: Mathematics']
)

import os
import sys
import subprocess

import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg: "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(PACKAGE_NAME)
    return config

def get_version():
    """Obtain the version number"""
    import imp
    mod = imp.load_source('version', os.path.join(PACKAGE_NAME, 'version.py'))
    return mod.__version__

# Documentation building command
try:
    from sphinx.setup_command import BuildDoc as SphinxBuildDoc
    class BuildDoc(SphinxBuildDoc):
        """Run in-place build before Sphinx doc build"""
        def run(self):
            ret = subprocess.call([sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building Scipy failed!")
            SphinxBuildDoc.run(self)
    cmdclass = {'build_sphinx': BuildDoc}
except ImportError:
    cmdclass = {}

# Call the setup function
if __name__ == "__main__":
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          include_package_data=True,
          test_suite="nose.collector",
          cmdclass=cmdclass,
          version=get_version(),
          **EXTRA_INFO)
