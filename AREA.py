import base_classifier as bc
import numpy as np
from keras.layers import Input, Dense
from keras import Model
import keras
from keras.optimizers import Adam
import tensorflow as tf
import datasets as ds


class AREA(bc.BaseClassifier):

    @staticmethod
    def the_loss(y_true, y_pred):

        # calcolo (y true servono solo come fattori)
        y_pred = y_pred * y_true[:, 0]
        y_pred = tf.maximum(0., 1. - y_pred)
        y_pred = y_pred * y_true[:, 1]
        y_pred = tf.reduce_sum(y_pred)

        return y_pred

    def get_grid_hyperparameters(self):
        return {
            'Learning_rate': np.logspace(-4, -1, 4),
            'Weight_decay': np.logspace(-5, -2, 4),
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

        product_function = []
        product_loss = []

        # fattori per gli esempi non etichettati (è importante come viene definito ds_all)
        for i in range(N_U):
            product_function.append([-1 if k != len(self.classes) - 1 else 1 for k, _ in enumerate(self.classes)])  # vettore di -1 o +1 alla fine
            product_loss.append([1. / N_U if k == len(self.classes) - 1 else 1 / (N_U * (K - 1)) for k, _ in enumerate(self.classes)])  #

        # fattori per gli esempi etichettati (vedere la definizione nel paper)
        for y in y_labeled:
            product_function.append([1 if k != len(self.classes) -1 else -1 for k, _ in enumerate(self.classes)]) # vettore di +1 o -1 alla fine

            p_l = (K / (K - 1)) * positive_class_factors[y]
            product_loss.append([p_l if k == y or k == len(self.classes) -1 else 0 for k, _ in enumerate(self.classes)]) #

        factors = []
        y_all_categorical = keras.utils.to_categorical(y_all, len(self.classes))

        for i in range(len(product_loss)):
            factors.append([product_function[i], product_loss[i], y_all_categorical[i]])
        factors = np.array(factors).reshape((len(product_loss), 3, K))

        # labels utilizzate per la validazione
        categ_y_test = np.array(
            [[[0 for _ in range(len(self.classes))], [0 for _ in range(len(self.classes))], x] for x in
             keras.utils.to_categorical(y_test, len(self.classes))]).reshape((len(x_test), 3, K))

        # allenamento col miglior modello trovato
        return model.fit(ds_all, factors, batch_size=256, epochs=200, shuffle=True,
                         validation_data=(x_test, categ_y_test),
                         verbose=0)
