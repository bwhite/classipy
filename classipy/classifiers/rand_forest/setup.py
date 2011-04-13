#!/usr/bin/env python
import sys
import re
import subprocess
from distutils.core import setup
from distutils.extension import Extension
import glob
import os


def get_cython_version():
    """
    Returns:
        Version as a pair of ints (major, minor)

    Raises:
        ImportError: Can't load cython or find version
    """
    import Cython.Compiler.Main
    match = re.search('^([0-9]+)\.([0-9]+)',
                      Cython.Compiler.Main.Version.version)
    try:
        return map(int, match.groups())
    except AttributeError:
        raise ImportError

# Only use Cython if it is available, else just use the pre-generated files
try:
    cython_version = get_cython_version()
    # Requires Cython version 0.13 and up
    if cython_version[0] == 0 and cython_version[1] < 13:
        raise ImportError
    from Cython.Distutils import build_ext
    source_ext = '.pyx'
    cmdclass = {'build_ext': build_ext}
except ImportError:
    source_ext = '.c'
    cmdclass = {}

ext_modules = [Extension("_rand_forest", ["rand_forest" + source_ext])]
setup(name='rand_forest',
      cmdclass=cmdclass,
      version='0.0.1',
      packages=[],
      package_data={},
      author='Brandyn A. White',
      author_email='bwhite@dappervision.com',
      license='GPL',
      url='https://github.com/bwhite/classipy',
      ext_modules=ext_modules)
