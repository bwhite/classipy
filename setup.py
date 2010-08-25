from distutils.core import setup
import subprocess

subprocess.check_call('scons')
setup(name='classipy',
      version='.01',
      packages=['classipy', 'classipy.classifiers', 'classipy.classifiers.libsvm', 'classipy.classifiers.liblinear'],
      package_data = {'classipy' : ['lib/*.so']}
      )
