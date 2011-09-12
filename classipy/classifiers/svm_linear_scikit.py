from classipy.classifiers.svm_scikit import SVMScikit


class SVMLinearScikit(SVMScikit):
    
    def __init__(self, **kw):
        super(SVMLinearScikit, self).__init__()
        self._param = kw
        self._labels = None

    def _make_model(self):
        from scikits.learn import svm
        kw = dict(self._param)
        self._m = svm.LinearSVC(**kw)
