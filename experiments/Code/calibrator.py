# borrowed from Turner (Uber Research https://github.com/uber-research/metropolis-hastings-gans/blob/master/mhgan/classification.py)
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class Calibrator(object):
    def fit(self, y_pred, y_true):
        raise NotImplementedError

    def predict(self, y_pred):
        raise NotImplementedError

    @staticmethod
    def validate(y_pred, y_true=None):
        y_pred = np.asarray(y_pred)
        assert y_pred.ndim == 1
        assert y_pred.dtype.kind == 'f'
        assert np.all(0 <= y_pred) and np.all(y_pred <= 1)

        if y_true is not None:
            y_true = np.asarray(y_true)
            assert y_true.shape == y_pred.shape
            assert y_true.dtype.kind == 'b'

        return y_pred, y_true

class Linear(Calibrator):
    def __init__(self):
        self.clf = LogisticRegression()

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        self.clf.fit(y_pred[:, None], y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_calib = self.clf.predict_proba(y_pred[:, None])[:, 1]
        return y_calib


class Isotonic(Calibrator):
    def __init__(self):
        self.clf = IsotonicRegression(y_min=0.0, y_max=1.0,
                                      out_of_bounds='clip')

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        self.clf.fit(y_pred, y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_calib = self.clf.predict(y_pred)
        return y_calib

class Beta1(Calibrator):
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.clf = LogisticRegression()

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        y_pred = logit(np.clip(y_pred, self.epsilon, 1.0 - self.epsilon))
        self.clf.fit(y_pred[:, None], y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_pred = logit(np.clip(y_pred, self.epsilon, 1.0 - self.epsilon))
        y_calib = self.clf.predict_proba(y_pred[:, None])[:, 1]
        return y_calib


class Beta2(Calibrator):
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.clf = LogisticRegression()

    def fit(self, y_pred, y_true):
        assert y_true is not None
        y_pred, y_true = Calibrator.validate(y_pred, y_true)
        y_pred = np.clip(y_pred.astype(np.float_),
                         self.epsilon, 1.0 - self.epsilon)
        y_pred = np.stack((np.log(y_pred), np.log(1.0 - y_pred)), axis=1)
        self.clf.fit(y_pred, y_true)

    def predict(self, y_pred):
        y_pred, _ = Calibrator.validate(y_pred)
        y_pred = np.clip(y_pred.astype(np.float_),
                         self.epsilon, 1.0 - self.epsilon)
        y_pred = np.stack((np.log(y_pred), np.log(1.0 - y_pred)), axis=1)
        y_calib = self.clf.predict_proba(y_pred)[:, 1]
        return y_calib