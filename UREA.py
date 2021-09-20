import base_classifier as bc
import numpy as np
from keras.layers import Input, Dense
from keras import Model
import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import datasets as ds
import class_prior_estimation


# Unbiased Risk EstimAtor taken from the paper 'Learning from Multi-Class Positive and Unlabeled Data'
# The class prior are not estimated but computed from all the labels of the dataset
class UREA(bc.BaseClassifier):

    @staticmethod
    def the_loss(y_true, y_pred):

        # compute hinge(z) for each class
        y_pred_pos = tf.maximum(0., 1. - y_pred)

        # comput hinge(-z) for each class
        y_pred_neg = tf.maximum(0., 1. - (-y_pred))

        res = y_pred_pos * y_true[:, 0]  # multiplication for hinge(z)
        res += y_pred_neg * y_true[:, 1]  # multiplication for hinge(-z)

        res = tf.reduce_sum(res)  # sum all addends

        return res

    def get_grid_hyperparameters(self):
        if self.validate_hyp:
            return {
                'Learning_rate': np.logspace(-4, -1, 4),
                'Weight_decay': np.logspace(-4, -1, 4),
            }
        else:
            return {
                'Learning_rate': [1e-3],
                'Weight_decay': [1e-3],
            }

    def get_model(self, input_dim, hyp):

        input = Input(shape=(input_dim,))

        # A matrix W and a bias to be learnt that maps the input space to the class space
        l = Dense(len(self.classes), activation=None, use_bias=True,
                  kernel_regularizer=keras.regularizers.l2(hyp['Weight_decay']),)(input)

        model = Model(input, l)

        model.compile(Adam(hyp['Learning_rate']), self.the_loss, metrics=[self.accuracy_metric])

        return model

    def accuracy_metric(self, y_true, y_pred):
        # the third item of each instance contains the true label
        return tf.metrics.categorical_accuracy(y_true[:, 2], y_pred)

    def predict(self, model, x):
        # The highest class score is chosen
        return np.argmax(model.predict(x), axis=1)

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):

        # Due to the loss formulation, it needs to compute loss factors for each instance, based on its label class
        K = len(self.classes)
        N_U = len(ds_unlabeled)

        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        # getting prior class probabilities (taken from the dataset itself)
        positive_class_factors = []
        for p_c in self.positive_classes:
            #els_class, _ = ds.filter_ds(ds_all, y_all, [p_c])
            #prior = len(els_class) / len(ds_all)

            els_class_labeled, _ = ds.filter_ds(ds_labeled, y_labeled, [p_c])

            prior = self.data_for_run['priors'][p_c]
            n_labeled = len(els_class_labeled)

            positive_class_factors.append(prior / n_labeled)

        # factors for the loss function
        product_loss_pos = []
        product_loss_neg = []

        # unlabeled samples factors
        for i in range(N_U):
            product_loss_pos.append([1. / N_U if k == K - 1 else 0 for k, _ in enumerate(self.classes)])
            product_loss_neg.append([0 if k == K - 1 else 1 / (N_U * (K - 1)) for k, _ in enumerate(self.classes)])  #

        # labeled samples factors
        for y in y_labeled:
            p_l = positive_class_factors[y] # prior / n_labeled
            product_loss_pos.append([p_l if k == y else -p_l if k == K - 1 else 0 for k, _ in enumerate(self.classes)])

            p_l = p_l / (K - 1)
            product_loss_neg.append([-p_l if k == y else p_l if k == K - 1 else 0 for k, _ in enumerate(self.classes)])

        # in this vector there are 3 items for each instance: the first one indicates the factor for hinge(z),
        # the second one indicates the factor for hinge(-z), the third indicates the categorical true label
        factors = []

        y_all_categorical = tf.keras.utils.to_categorical(y_all, K)  # true labels

        for i in range(len(ds_all)):
            factors.append([product_loss_pos[i], product_loss_neg[i], y_all_categorical[i]])
        factors = np.array(factors).reshape((len(ds_all), 3, K))

        # labels used for test set (no need for factors)
        categ_y_test = np.array([[[0 for _ in range(K)], [0 for _ in range(K)], x]
                                 for x in tf.keras.utils.to_categorical(y_test, K)
                                 ]).reshape((len(x_test), 3, K))

        # train model
        return model.fit(ds_all, factors, batch_size=256, epochs=200, shuffle=True,
                         validation_data=(x_test, categ_y_test), verbose=0)

    def run_preparation(self, ds_labeled, y_labeled, ds_unlabeled):

        # merge all positive class in one class
        data = np.concatenate((ds_labeled, ds_unlabeled))
        labels = np.concatenate(([1 for _ in y_labeled], [0 for _ in ds_unlabeled]))

        shuffl = np.random.permutation(len(data))
        data = data[shuffl]
        labels = labels[shuffl]

        # normalization
        data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

        # compute the prior for the positive super class
        alfa = class_prior_estimation.get_prior(data, labels)

        # compute each class prior as a fraction of the total positive class prior
        self.data_for_run['priors'] = [alfa * len([1 for sub_y in y_labeled if sub_y == y]) / len(ds_labeled) for y
                                       in np.unique(y_labeled)]