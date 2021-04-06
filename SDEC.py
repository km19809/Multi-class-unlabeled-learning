import base_classifier as bc
import numpy as np
import keras
from keras import Model, Input
from keras.layers import Dense, Layer, InputSpec
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
import datasets as ds
import tensorflow.keras.backend as K
import gc
from keras.callbacks import History


class SDEC(bc.BaseClassifier):

    def get_sup_loss(self, beta_same=1., beta_diff=1.):
        def my_sup_loss(y_true, y_pred):

            # calcolo coefficienti y
            y_same = tf.matmul(y_true, tf.transpose(y_true))
            y_diff = (y_same - 1.) * -1.

            # calcolo distanze
            r = tf.reduce_sum(y_pred * y_pred, 1)
            r = tf.reshape(r, [-1, 1])
            D = r - 2 * tf.matmul(y_pred, tf.transpose(y_pred)) + tf.transpose(r)
            distances = D

            # calcolo loss per quelli della stessa classe
            final = y_same * tf.maximum(0., distances - beta_same)

            # calcolo loss per quelli di classe differente
            final += y_diff * tf.maximum(0., beta_diff - distances)

            # viene presa la parte superiore della matrice quadrata
            final = tf.linalg.band_part(final, 0, -1)
            final = tf.reduce_sum(final)

            # normalizzazione in base al numero di elementi
            n_elements = tf.cast(tf.shape(y_pred)[0], 'float32')
            return final / ((n_elements ** 2 - n_elements) / 2)

        return my_sup_loss

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def get_model(self, input_dim, hyp):

        act = 'relu'
        init = 'glorot_uniform'
        # no weight regulizer
        w_dec = 1e-4

        # LAYERS
        dims = [input_dim, 500, 500, 2000, 10]

        n_stacks = len(dims) - 1

        input_data = Input(shape=(dims[0],), name='input')
        x = input_data

        # internal layers of encoder
        for i in range(n_stacks - 1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i,
                      kernel_regularizer=keras.regularizers.l2(w_dec))(x)

        # latent hidden layer
        encoded = Dense(dims[-1], activation='linear', kernel_initializer=init, name='encoder',
                        kernel_regularizer=keras.regularizers.l2(w_dec))(x)

        # internal layers of decoder
        x = encoded
        for i in range(n_stacks - 1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i,
                      kernel_regularizer=keras.regularizers.l2(w_dec))(x)

        # decoder output
        x = Dense(dims[0], kernel_initializer=init, name='decoder',)(x)
        decoded = x

        return self.set_model_output(input_data, encoded, decoded, hyp)

    def set_model_output(self, input_data, encoded, decoded, hyp, include_clustering=False, centroids=None):

        gamma_kld = 0.1

        # supervised loss
        loss_labeled = ['mse', self.get_sup_loss(hyp["Beta_sup"], hyp["Beta_sup"])]
        output_labeled = [decoded, encoded]
        loss_weights_labeled = [1, hyp['Gamma_sup']]

        # unsupervised loss
        loss_unlabeled = ['mse']
        output_unlabeled = [decoded]
        loss_weights_unlabeled = [1.]

        if include_clustering:
            unlabeled_last_layer = ClusteringLayer(len(self.classes), weights=[centroids], name='clustering')(encoded)

            output_labeled.append(unlabeled_last_layer)
            output_unlabeled.append(unlabeled_last_layer)

            loss_weights_labeled.append(gamma_kld)
            loss_weights_unlabeled.append(gamma_kld)

            loss_labeled.append('kld')
            loss_unlabeled.append('kld')

        # define models
        model_unlabeled = Model(inputs=input_data, outputs=output_unlabeled, name="unlabeled")
        model_labeled = Model(inputs=input_data, outputs=output_labeled, name="labeled")

        # compile models
        model_unlabeled.compile(loss=loss_unlabeled, loss_weights=loss_weights_unlabeled, optimizer=Adam())
        model_labeled.compile(loss=loss_labeled, loss_weights=loss_weights_labeled, optimizer=Adam())

        return model_unlabeled, model_labeled  # MIND THE ORDER

    def get_grid_hyperparameters(self):
        # no learning parameter, weight decay
        return {
            'Beta_sup': np.logspace(0, 3, 4), # float
            'Gamma_sup': np.logspace(-2, 1, 4), # float
            #'Embedding_dim': np.linspace(6, 18, 4)  # int

            #'Beta_sup':  np.logspace(2, 2, 1),  # float
            #'Gamma_sup': np.logspace(-2, -2, 1),  # float
            #'Embedding_dim': np.linspace(6, 18, 1) - 1  # int
        }

    def predict(self, model, x):
        model_unlabeled = model[0]

        # nearest centroid
        return model_unlabeled.predict(x)[1].argmax(1)

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):
        epochs_pretraining = 150
        max_iter_clustering = 10000

        model_unlabeled, model_labeled = model
        input_data = model_unlabeled.layers[0].input
        encoded = model_unlabeled.get_layer("encoder").output
        decoded = model_unlabeled.get_layer("decoder").output

        # first train
        history_pre = self.run_pretraining(model_unlabeled, model_labeled, ds_labeled, y_labeled, ds_unlabeled, epochs_pretraining)

        # kmeans for centroids
        centroids = self.get_centroids_from_kmeans(model_labeled, ds_unlabeled, ds_labeled, y_labeled)

        # models for the second step
        model_unlabeled, model_labeled = self.set_model_output(input_data, encoded, decoded, current_hyp, True, centroids)
        model = (model_unlabeled, model_labeled)

        history_clu, epochs_clu, data_plot = self.run_clustering(model_unlabeled, model_labeled,
                                                                 ds_labeled, y_labeled, ds_unlabeled, y_unlabeled,
                                                                 x_test, y_test, max_iter_clustering)

        # set history object
        history = History()
        history.epoch = [i for i in range(epochs_pretraining)] + [i + epochs_pretraining for i in range(epochs_clu)]
        history.epoch_acc = [i + epochs_pretraining for i in range(epochs_clu)]
        history.history = {
            "loss_rec": history_pre["loss_rec"] + history_clu["loss_rec"],
            "loss_sup": history_pre["loss_sup"] + history_clu["loss_sup"],
            "loss_clu": [0. for _ in range(epochs_pretraining)] + history_clu["loss_clu"],
            "accuracy_metric": history_clu["accuracy_metric"],
            "val_accuracy_metric": history_clu["val_accuracy_metric"],
        }
        history.data_plot = data_plot

        return model, history

    def run_pretraining(self, model_unlabeled, model_labeled, ds_labeled,
                        y_labeled_original, ds_unlabeled, epochs_pretraining):

        history = dict()
        history["loss_rec"] = []
        history["loss_sup"] = []

        y_labeled = keras.utils.to_categorical(y_labeled_original, len(self.classes))

        bs_unlab = 256
        bs_lab = 256
        epoch = 0

        while epoch < epochs_pretraining:
            # print("EPOCH {}".format(epoch))

            if epoch % 50 == 0:
                gc.collect()

            # shuffle labeled  and unlabeled dataset
            shuffler_l = np.random.permutation(len(ds_labeled))
            ds_labeled = ds_labeled[shuffler_l]
            y_labeled = y_labeled[shuffler_l]

            shuffler_l = np.random.permutation(len(ds_unlabeled))
            ds_unlabeled = ds_unlabeled[shuffler_l]

            # variabili per l'epoca
            i_unlab = 0
            i_lab = 0

            finish_labeled = False
            finish_unlabeled = False
            epoch_losses = []

            while not (finish_labeled and finish_unlabeled):

                # unlabeled train
                if not finish_unlabeled:
                    if (i_unlab + 1) * bs_unlab >= ds_unlabeled.shape[0]:
                        t_unlabeled = [ds_unlabeled[i_unlab * bs_unlab::]]
                        b_unlabeled = ds_unlabeled[i_unlab * bs_unlab::]

                        finish_unlabeled = True
                    else:
                        t_unlabeled = [ds_unlabeled[i_unlab * bs_unlab:(i_unlab + 1) * bs_unlab]]
                        b_unlabeled = ds_unlabeled[i_unlab * bs_unlab:(i_unlab + 1) * bs_unlab]

                        i_unlab += 1

                    losses = model_unlabeled.train_on_batch(b_unlabeled, t_unlabeled)
                    epoch_losses.append([losses, 0.])

                # labeled train
                if not finish_labeled:
                    if (i_lab + 1) * bs_lab >= ds_labeled.shape[0]:

                        t_labeled = [ds_labeled[i_lab * bs_lab::], y_labeled[i_lab * bs_lab::]]
                        b_labeled = ds_labeled[i_lab * bs_lab::]

                        finish_labeled = True
                    else:
                        t_labeled = [ds_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab], y_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab]]
                        b_labeled = ds_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab]

                        i_lab += 1

                    losses = model_labeled.train_on_batch(b_labeled, t_labeled)
                    epoch_losses.append(losses[1:])

            # calcolo loss per l'epoca
            losses = np.sum(epoch_losses, axis=0)
            history["loss_rec"].append(losses[0])
            history["loss_sup"].append(losses[1])

            epoch += 1

        return history

    def run_clustering(self, model_unlabeled, model_labeled, ds_labeled_original, y_labeled_original, ds_unlabeled_original, y_unlabeled,
                       x_test, y_test, maxiter):

        history = dict()
        history["loss_rec"] = []
        history["loss_sup"] = []
        history["loss_clu"] = []
        history["accuracy_metric"] = []
        history["val_accuracy_metric"] = []

        # tolerance threshold to stop training
        tol = 0.001
        bs_unlab = 256
        bs_lab = 256

        # ottenimento ds completo
        all_x = np.concatenate((ds_labeled_original, ds_unlabeled_original), axis=0)
        all_y = np.concatenate((y_labeled_original, y_unlabeled), axis=0)
        labeled_indexes = np.array([i < len(ds_labeled_original) for i, _ in enumerate(all_x)])
        unlabeled_indexes = np.array([i >= len(ds_labeled_original) for i, _ in enumerate(all_x)])

        y_labeled_cat_original = keras.utils.to_categorical(y_labeled_original, len(self.classes))

        y_pred_last = None
        p = None
        stop_for_delta = False
        batch_n = 0
        epoch = 0
        plot_interval = 100
        clustering_data_plot = dict()

        while batch_n < maxiter and not stop_for_delta:

            # Memorizzazione stato dei cluster
            if epoch % plot_interval == 0:
                clustering_data_plot[epoch] = {
                    'centroids': model_unlabeled.get_layer('clustering').get_centroids(),
                    'x_data': model_labeled.predict(all_x)[1],
                    'y_data': all_y,
                    'y_pred': self.predict((model_unlabeled,), all_x),
                    'lab_index': labeled_indexes
                }

            # print("EPOCH {}, Batch n° {}".format(epoch, batch_n))
            if epoch % 50 == 0:
                gc.collect()

            # shuffle labeled dataset
            shuffler_l = np.random.permutation(len(ds_labeled_original))
            ds_labeled = ds_labeled_original[shuffler_l]
            y_labeled = y_labeled_cat_original[shuffler_l]

            # unlabeled dataset
            shuffler_u = np.random.permutation(len(ds_unlabeled_original))
            ds_unlabeled = ds_unlabeled_original[shuffler_u]

            if p is not None:
                p_lab = p[labeled_indexes][shuffler_l]
                p_unlab = p[unlabeled_indexes][shuffler_u]

            i_unlab = 0
            i_lab = 0
            ite = 0
            epoch_losses = [[0., 0., 0.]]

            finish_labeled = False
            finish_unlabeled = False

            while not (finish_labeled and finish_unlabeled):

                # update target probability
                if batch_n % self.update_interval == 0:

                    # PREDICT cluster probabilities
                    q = model_unlabeled.predict(all_x)[1]

                    # check stop criterion
                    y_pred_new = q.argmax(1)
                    if y_pred_last is not None:
                        delta_label = sum(y_pred_new[i] != y_pred_last[i] for i in range(len(y_pred_new))) / y_pred_new.shape[0]
                        if delta_label < tol:
                            print('Reached stopping criterium, delta_label ', delta_label, '< tol ', tol, '. Iter n°', batch_n)
                            stop_for_delta = True
                            break

                    # set new predicted labels
                    y_pred_last = y_pred_new

                    # update the auxiliary target distribution p
                    p = self.target_distribution(q)
                    p_lab = p[labeled_indexes][shuffler_l]
                    p_unlab = p[unlabeled_indexes][shuffler_u]

                # unlabeled train
                if not finish_unlabeled:
                    if (i_unlab + 1) * bs_unlab >= ds_unlabeled.shape[0]:
                        t_unlabeled = [ds_unlabeled[i_unlab * bs_unlab::], p_unlab[i_unlab * bs_unlab::]]
                        b_unlabeled = ds_unlabeled[i_unlab * bs_unlab::]

                        finish_unlabeled = True
                    else:
                        t_unlabeled = [ds_unlabeled[i_unlab * bs_unlab:(i_unlab + 1) * bs_unlab],
                                       p_unlab[i_unlab * bs_unlab:(i_unlab + 1) * bs_unlab]]

                        b_unlabeled = ds_unlabeled[i_unlab * bs_unlab:(i_unlab + 1) * bs_unlab]

                        i_unlab += 1

                    losses = model_unlabeled.train_on_batch(b_unlabeled, t_unlabeled)
                    epoch_losses.append([losses[1], 0., losses[2]])

                    batch_n += 1

                # labeled train
                if not finish_labeled:
                    if (i_lab + 1) * bs_lab >= ds_labeled.shape[0]:
                        t_labeled = [ds_labeled[i_lab * bs_lab::], y_labeled[i_lab * bs_lab::], p_lab[i_lab * bs_lab::]]
                        b_labeled = ds_labeled[i_lab * bs_lab::]

                        finish_labeled = True
                    else:
                        t_labeled = [ds_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab],
                                     y_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab],
                                     p_lab[i_lab * bs_lab:(i_lab + 1) * bs_lab]]

                        b_labeled = ds_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab]

                        i_lab += 1

                    losses = model_labeled.train_on_batch(b_labeled, t_labeled)
                    epoch_losses.append(losses[1:])

                    batch_n += 1

                ite += 1

            # calcolo loss per l'epoca
            losses = np.sum(epoch_losses, axis=0)
            history["loss_rec"].append(losses[0])
            history["loss_sup"].append(losses[1])
            history["loss_clu"].append(losses[2])

            # accuracies
            history["accuracy_metric"].append(self.get_accuracy(self.predict((model_unlabeled,), all_x), all_y))
            history["val_accuracy_metric"].append(self.get_accuracy(self.predict((model_unlabeled,), x_test), y_test))

            epoch += 1

        # stato finale dei cluster
        clustering_data_plot[epoch] = {
            'centroids': model_unlabeled.get_layer('clustering').get_centroids(),
            'x_data': model_labeled.predict(all_x)[1],
            'y_data': all_y,
            'y_pred': self.predict((model_unlabeled,), all_x),
            'lab_index': labeled_indexes
        }

        return history, epoch, clustering_data_plot

    def accuracy_metric(self, y_true, y_pred):
        raise Exception("Not implemented")
        #return tf.py_function(func=self.get_accuracy, inp=[y_pred, y_true], Tout=[tf.float32])

    @staticmethod
    def get_accuracy(y_pred, y_true):
        # cluster accuracy
        y_true1 = y_true.astype(np.int64)
        D = max(y_pred.max(), y_true1.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true1[i]] += 1

        row, col = linear_assignment(w.max() - w)

        acc = sum([a for a in w[row, col]]) * 1.0 / y_pred.size

        return acc

    def get_centroids_from_kmeans(self, model_labeled, x_unlabeled, x_labeled, y):

        if len(x_labeled) > 0:
            all_ds = np.concatenate((x_labeled, x_unlabeled), axis=0)
        else:
            all_ds = x_unlabeled

        all_x_encoded = model_labeled.predict(all_ds)[1]

        # ottenimento centroidi iniziali per le classi positive
        centroids = []
        if len(x_labeled) > 0:
            x_labeled_encoded = model_labeled.predict(x_labeled)[1]
            for y_class in self.positive_classes:
                only_x_class, _ = ds.filter_ds(x_labeled_encoded, y, [y_class])
                if len(only_x_class) > 0:
                    centroids.append(np.mean(only_x_class, axis=0))

        centroids = np.array(centroids)

        # si avviano diverse istanze di K-means
        best_kmeans = None

        for i in range(len(self.classes)):

            # si aggiungono dei centroidi per le classi negative. Esse sono centrate e hanno una scala di riferimento
            try_centroids = self.get_centroids_for_clustering(all_x_encoded, len(self.classes), centroids)

            kmeans = KMeans(n_clusters=len(self.classes), init=try_centroids, n_init=1)
            kmeans.fit(all_x_encoded)

            if best_kmeans is None or kmeans.inertia_ < best_kmeans.inertia_:
                best_kmeans = kmeans

        return best_kmeans.cluster_centers_

    def get_centroids_for_clustering(self, X, k, centers=None, pdf_method=True):

        # Sample the first point
        if centers is None or len(centers) == 0:
            initial_index = np.random.choice(range(X.shape[0]), )
            centers = np.array([X[initial_index, :].tolist()])

        while len(centers) < k:
            distance = self.dist(X, np.array(centers))

            if len(centers) == 0:
                pdf = distance / np.sum(distance)
                centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf.flatten())]
            else:
                # Calculate the distance of each point from its nearest centroid
                dist_min = np.min(distance, axis=1)
                if pdf_method:
                    pdf = dist_min / np.sum(dist_min)
                    # Sample one point from the given distribution
                    centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf)]
                else:
                    index_max = np.argmax(dist_min, axis=0)
                    centroid_new = X[index_max, :]

            centers = np.concatenate((centers, [centroid_new.tolist()]), axis=0)

        return np.array(centers)

    def dist(self, data, centers):
        distance = np.sum((np.array(centers) - data[:, None, :]) ** 2, axis=2)
        return distance


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim),
                                          initializer='glorot_uniform')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))

        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_centroids(self):
        return self.clusters.numpy()



