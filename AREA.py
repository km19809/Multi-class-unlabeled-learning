import base_classifier as bc
import numpy as np
from keras.layers import Input, Dense
from keras import Model
import keras
from keras.optimizers import Adam
import tensorflow as tf
import datasets as ds


# Alternative Risk EstimAtor taken from the paper 'Learning from Multi-Class Positive and Unlabeled Data'
# The class prior are not estimated but computed from all the labels of the dataset
class AREA(bc.BaseClassifier):

    @staticmethod
    def the_loss(y_true, y_pred):

        res = y_pred * y_true[:, 0]  # multiply argument for hinge function
        res = tf.maximum(0., 1. - res)  # compute hinge function
        res = res * y_true[:, 1]  # multiply function for a given factor
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
            els_class, _ = ds.filter_ds(ds_all, y_all, [p_c])
            els_class_labeled, _ = ds.filter_ds(ds_labeled, y_labeled, [p_c])

            prior = len(els_class) / len(ds_all)
            n_labeled = len(els_class_labeled)

            positive_class_factors.append(prior / n_labeled)

        # factors for the loss function
        product_argument = []
        product_loss = []

        # unlabeled samples factors
        for i in range(N_U):
            product_argument.append([-1 if k != K - 1 else 1 for k, _ in enumerate(self.classes)])
            product_loss.append([1. / N_U if k == K - 1 else 1 / (N_U * (K - 1)) for k, _ in enumerate(self.classes)])

        # labeled samples factors
        for y in y_labeled:
            product_argument.append([1 if k != K - 1 else -1 for k, _ in enumerate(self.classes)])

            p_l = (K / (K - 1)) * positive_class_factors[y]
            product_loss.append([p_l if k == y or k == K - 1 else 0 for k, _ in enumerate(self.classes)])

        # in this vector there are 3 items for each instance: the first one indicates the factor for the hinge argument,
        # the second one indicates the factor for the hinge function, the third indicates the categorical true label
        factors = []

        y_all_categorical = keras.utils.to_categorical(y_all, K)  # true labels

        for i in range(len(product_loss)):
            factors.append([product_argument[i], product_loss[i], y_all_categorical[i]])
        factors = np.array(factors).reshape((len(product_loss), 3, K))

        # labels used for test set (no need for factors)
        categ_y_test = np.array([[[0 for _ in range(K)], [0 for _ in range(K)], x]
                                 for x in keras.utils.to_categorical(y_test, K)
                                 ]).reshape((len(x_test), 3, K))

        # training model
        return model.fit(ds_all, factors, batch_size=256, epochs=200, shuffle=True,
                         validation_data=(x_test, categ_y_test), verbose=0)

