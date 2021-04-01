import base_classifier as bc
from sklearn.svm import LinearSVC, SVC
import numpy as np


class LinearSVM(bc.BaseClassifier):

    def get_model(self, input_dim, hyp):

        model = LinearSVC(dual=False, C=hyp['C'])

        return model

    def get_grid_hyperparameters(self):
        return {
            'C': np.logspace(-3, 10, 14),
        }

    def predict(self, model, x):
        return model.predict(x)

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):
        model.fit(ds_labeled, y_labeled)
        return None

    def accuracy_metric(self, y_true, y_pred):
        pass


class RbfSVM(bc.BaseClassifier):

    def get_model(self, input_dim, hyp):

        model = SVC(kernel='rbf', C=hyp['C'], gamma=hyp['Gamma'])

        return model

    def get_grid_hyperparameters(self):
        return {
            'C': np.logspace(-2, 10, 13),
            'Gamma': np.logspace(-9, 3, 12),
        }

    def predict(self, model, x):
        return model.predict(x)

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):
        model.fit(ds_labeled, y_labeled)
        return None

    def accuracy_metric(self, y_true, y_pred):
        pass