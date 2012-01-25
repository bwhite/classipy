from validation import cross_validation, evaluate, confidence_stats, confusion_stats, gen_confusion, select_parameters, whiten
from classifiers.base import BinaryClassifier
from classifiers.svm import SVM
from classifiers.svm_linear import SVMLinear
from classifiers.knn import KNN
from classifiers.lda import LDA
from _classipy_rand_forest import RandomForestClassifier
import _classipy_rand_forest as rand_forest
#from visualization import decision_boundary_2d
import _classipy_kernels as kernels
import pkg_resources as _pkg_resources  # NOTE(brandyn): Used to hint pyinstaller that we need it
