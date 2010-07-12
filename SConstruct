import distutils.sysconfig
env = Environment()
env.Replace(CXX = 'g++')
env.Append(CCFLAGS =  '-O3 -Wall -fPIC')

env.SharedLibrary('classipy/classifiers/libsvm/libsvm', ['thirdparty/libsvm-2.91/svm.cpp'])
env.SharedLibrary('classipy/classifiers/liblinear/liblinear', ['thirdparty/liblinear-1.6/linear.cpp', 'thirdparty/liblinear-1.6/tron.cpp', 'thirdparty/liblinear-1.6/blas/dnrm2.c', 'thirdparty/liblinear-1.6/blas/daxpy.c', 'thirdparty/liblinear-1.6/blas/ddot.c', 'thirdparty/liblinear-1.6/blas/dscal.c'])