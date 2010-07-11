import distutils.sysconfig
env = Environment()
env.Replace(CXX = 'g++')
env.Append(CCFLAGS =  '-O3 -Wall -fPIC')

env.SharedLibrary('classipy/classifiers/libsvm/libsvm', ['thirdparty/libsvm-2.91/svm.cpp'])
env.SharedLibrary('classipy/classifiers/liblinear/liblinear', ['thirdparty/libsvm-2.91/svm.cpp'])
lib: linear.o tron.o blas/blas.a
	$(CXX) -shared -dynamiclib linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)
