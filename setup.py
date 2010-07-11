from distutils.core import setup

setup(name='classipy',
      version='.01',
      packages=['classipy', 'classipy.classifiers', 'classipy.classifiers.libsvm', 'classipy.classifiers.liblinear'],
      package_data = {'classipy' : ['classifiers/libsvm/*.so', 'classifiers/liblinear/*.so']}
      )
