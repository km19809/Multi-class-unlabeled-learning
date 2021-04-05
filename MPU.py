import base_classifier as bc
import numpy as np
from keras.layers import Input, Dense
from keras import Model
import keras
from keras.optimizers import Adam
import tensorflow as tf
import datasets as ds
import math


class MPU(bc.BaseClassifier):

    def the_loss(self, y_true, y_pred):

        true_moltiplicator = y_true[:, 2 + len(self.classes):]
        vector_true = tf.reduce_sum(y_pred * true_moltiplicator, 1)

        # calcolo per parte labeled
        k_pred = y_pred[:, -1]  # parte k-esima predetta
        loss = y_true[:, 0] * tf.maximum(0., k_pred - vector_true)
        loss = tf.reduce_sum(loss)

        # parte unlabeled
        loss_unlab = tf.transpose(y_pred[:, :-1])  # parte predetta per le label
        loss_unlab = tf.maximum(0., 1. + loss_unlab - vector_true) * y_true[:, 1]

        loss += tf.reduce_sum(loss_unlab)

        return loss

    def get_encoding_labels(self):
        labels = []

        k = len(self.classes)
        r = k - 1

        a = (1 + math.sqrt(r + 1)) / r
        center = (a + 1) / (r + 1)

        for v in range(k - 1):
            vector = []
            for p in range(r):
                point = (0 if p != v else 1) - center
                vector.append(point)

            labels.append(vector)

        labels.append([a - center for _ in range(r)])

        # scale data for unit distance from origin
        scale = math.sqrt(sum([d ** 2 for d in labels[0]]))
        labels = [[p / scale for p in v] for v in labels]

        return labels

    def get_pseudolabels(self, ds_unlabeled, model):
        predictions = model.predict(ds_unlabeled)

        pseudo_labels = []
        for pred in predictions:
            min_y = None
            best_loss = None
            for y in range(len(self.classes)):
                loss = sum([max(0., 1. + p - pred[y]) for p in pred[:-1]])

                if min_y is None or loss < best_loss:
                    min_y = y
                    best_loss = loss

            pseudo_labels.append(min_y)

        return np.array(pseudo_labels)

    def get_grid_hyperparameters(self):
        return {
            'Learning_rate': np.logspace(-5, -1, 5),
            'Weight_decay': np.logspace(-5, -1, 5),
        }

    def get_model(self, input_dim, hyp):
        # I bias non vengono utilizzati

        dims = [len(self.classes) - 1, len(self.classes)]

        input = Input(shape=(input_dim,))
        l = input

        for i in range(len(dims)):
            act = 'relu' if i < len(dims) - 2 else None
            trainable = i < len(dims) - 1

            l = Dense(dims[i], activation=act, use_bias=False, trainable=trainable,
                      kernel_regularizer=keras.regularizers.l2(hyp['Weight_decay']),)(l)

        model = Model(input, l)
        model.compile(Adam(hyp['Learning_rate']), self.the_loss, metrics=[self.accuracy_metric])

        # pesi per l'ultimo layer (embeddings)
        embeddings = np.array(self.get_encoding_labels()).transpose()
        w_last = model.layers[-1].get_weights()
        w_last[0] = embeddings
        model.layers[-1].set_weights(w_last)

        return model

    def predict(self, model, x):
        return np.argmax(model.predict(x), axis=1)

    def accuracy_metric(self, y_true, y_pred):
        return tf.metrics.categorical_accuracy(y_true[:, 2:2 + len(self.classes)], y_pred)

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):

        # determinazione dei fattori da utilizzare per la loss
        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        # determinazione dei fattori da utilizzare per la loss
        K = len(self.classes)
        N_U = len(ds_unlabeled)

        # ottenimento probabilità a priori (per ora vengono lette dal dataset)
        positive_class_factors = []
        for p_c in self.positive_classes:
            els_class, _ = ds.filter_ds(ds_all, y_all, [p_c])
            els_class_labeled, _ = ds.filter_ds(ds_labeled, y_labeled, [p_c])

            prior = len(els_class) / len(ds_all)
            n_labeled = len(els_class_labeled)

            positive_class_factors.append(prior / n_labeled)

        product_loss_lab = []
        product_loss_unlab = []

        # fattori per gli esempi non etichettati (è importante come viene definito ds_all)
        for i in range(N_U):
            product_loss_lab.append(0.)
            product_loss_unlab.append(1 / (2 * N_U * (K - 1)))

        # fattori per gli esempi etichettati (vedere la definizione nel paper)
        for y in y_labeled:
            product_loss_lab.append(positive_class_factors[y] / (2 * (K - 1)))
            product_loss_unlab.append(0.)

        tol = 0.001
        epochs_per_iter = 100 # todo quale valore specificare?
        max_iter = 100

        old_pseudo_y_unlab = None
        iter = 0
        history = None

        while iter < max_iter:

            # get argmax labels
            pseudo_y_unlab = self.get_pseudolabels(ds_unlabeled, model)

            # convergence criterium
            if old_pseudo_y_unlab is not None:
                delta_label = sum(pseudo_y_unlab[i] != old_pseudo_y_unlab[i] for i in range(len(pseudo_y_unlab))) / pseudo_y_unlab.shape[0]
                if delta_label < tol:
                    print('Reached stopping criterium, delta_label ', delta_label, '< tol ', tol)
                    break

            old_pseudo_y_unlab = pseudo_y_unlab

            pseudo_y_all = np.concatenate((pseudo_y_unlab, y_labeled), axis=0)

            y_all_categorical = keras.utils.to_categorical(y_all, len(self.classes))

            # si calcolano i fattori da dare in pasto alla loss
            factors = [[product_loss_lab[i]] + [product_loss_unlab[i]] + [x for x in y_all_categorical[i]] + [0 if k != pseudo_y_all[i] else 1 for k in range(K)]
                for i in range(len(ds_all))]
            factors = np.array(factors)

            # labels utilizzate per la validazione
            categ_y_test = np.array([[0, 0, ] + [xx for xx in x] + [0 for k in range(K)] for x in keras.utils.to_categorical(y_test, len(self.classes))])

            # train parameters
            _history = model.fit(ds_all, factors, batch_size=256, epochs=epochs_per_iter, shuffle=True,
                          validation_data=(x_test, categ_y_test),
                          verbose=0)

            # si mantiene la storia di tutte le epoche
            if history is None:
                history = _history
            else:
                history.epoch.extend([x + iter * epochs_per_iter for x in _history.epoch])
                for key in _history.history.keys():
                    value = _history.history[key]
                    history.history[key].extend(value)

            iter += 1

        return history
