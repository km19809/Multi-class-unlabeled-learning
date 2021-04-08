import base_classifier as bc
import numpy as np
from keras.layers import Input, Dense
from keras import Model
import keras
from keras.optimizers import Adam
import tensorflow as tf
import datasets as ds


class UREA(bc.BaseClassifier):

    @staticmethod
    def the_loss(y_true, y_pred):

        y_pred_pos = tf.maximum(0., 1. - y_pred)
        y_pred_neg = tf.maximum(0., 1. - (-y_pred))

        res = y_pred_pos * y_true[:, 0]  # calcolo per hinge(z)
        res += y_pred_neg * y_true[:, 1]  # calcolo per hinge(-z)

        res = tf.reduce_sum(res)

        return res

    def get_grid_hyperparameters(self):
        return {
            'Learning_rate': np.logspace(-5, -1, 5),
            'Weight_decay': np.logspace(-5, -1, 5),
        }

    def get_model(self, input_dim, hyp):
        dims = [len(self.classes)]

        input = Input(shape=(input_dim,))
        l = input

        for i in range(len(dims)):
            act = 'relu' if i != len(dims) - 1 else None
            l = Dense(dims[i], activation=act,
                      kernel_regularizer=keras.regularizers.l2(hyp['Weight_decay']),)(l)

        model = Model(input, l)
        model.compile(Adam(hyp['Learning_rate']), self.the_loss, metrics=[self.accuracy_metric])

        return model

    def accuracy_metric(self, y_true, y_pred):
        return tf.metrics.categorical_accuracy(y_true[:, 2], y_pred)

    def predict(self, model, x):
        return np.argmax(model.predict(x), axis=1)

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):

        # determinazione dei fattori da utilizzare per la loss
        K = len(self.classes)
        N_U = len(ds_unlabeled)

        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        # ottenimento probabilità a priori (per ora vengono lette dal dataset)
        positive_class_factors = []
        for p_c in self.positive_classes:
            els_class, _ = ds.filter_ds(ds_all, y_all, [p_c])
            els_class_labeled, _ = ds.filter_ds(ds_labeled, y_labeled, [p_c])

            prior = len(els_class) / len(ds_all)
            n_labeled = len(els_class_labeled)

            positive_class_factors.append(prior / n_labeled)

        product_loss_pos = []
        product_loss_neg = []

        # fattori per gli esempi non etichettati (è importante come viene definito ds_all)
        for i in range(N_U):
            product_loss_pos.append(
                [1. / N_U if k == len(self.classes) - 1 else 0 for k, _ in enumerate(self.classes)])  #
            product_loss_neg.append(
                [0 if k == len(self.classes) - 1 else 1 / (N_U * (K - 1)) for k, _ in enumerate(self.classes)])  #

        # fattori per gli esempi etichettati (vedere la definizione nel paper)
        for y in y_labeled:
            p_l = positive_class_factors[y]
            product_loss_pos.append(
                [p_l if k == y else -p_l if k == len(self.classes) - 1 else 0 for k, _ in enumerate(self.classes)])

            p_l = p_l / (K - 1)
            product_loss_neg.append(
                [-p_l if k == y else p_l if k == len(self.classes) - 1 else 0 for k, _ in enumerate(self.classes)])

        factors = []
        y_all_categorical = keras.utils.to_categorical(y_all, len(self.classes))

        for i in range(len(product_loss_pos)):
            factors.append([product_loss_pos[i], product_loss_neg[i], y_all_categorical[i]])
        factors = np.array(factors).reshape((len(product_loss_pos), 3, K))

        # labels utilizzate per la validazione
        categ_y_test = np.array([[[0 for _ in range(len(self.classes))], [0 for _ in range(len(self.classes))], x] for x in keras.utils.to_categorical(y_test, len(self.classes))]).reshape((len(x_test), 3, K))

        # train
        return model.fit(ds_all, factors, batch_size=256, epochs=200, shuffle=True,
                         validation_data=(x_test, categ_y_test),
                         verbose=0)