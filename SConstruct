import distutils.sysconfig
env = Environment()
env.Replace(CXX = 'g++')
env.Append(CCFLAGS =  '-O3 -Wall -fPIC')

env.SharedLibrary('classipy/lib/libsvm', ['thirdparty/libsvm-3.1/svm.cpp'])
env.SharedLibrary('classipy/lib/liblinear', ['thirdparty/liblinear-1.6/linear.cpp', 'thirdparty/liblinear-1.6/tron.cpp', 'thirdparty/liblinear-1.6/blas/dnrm2.c', 'thirdparty/liblinear-1.6/blas/daxpy.c', 'thirdparty/liblinear-1.6/blas/ddot.c', 'thirdparty/liblinear-1.6/blas/dscal.c'])