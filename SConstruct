import distutils.sysconfig
env = Environment()
env.Replace(CXX = 'g++')
env.Append(CCFLAGS =  '-O3 -Wall -fPIC')

env.SharedLibrary('classipy/classifiers/libsvm/libsvm', ['thirdparty/libsvm-2.91/svm.cpp'])