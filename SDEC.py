import base_classifier as bc
import numpy as np
import keras
from keras import Model, Input
from keras.layers import Dense, Layer, InputSpec
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
import datasets as ds
import tensorflow.keras.backend as K


class SDEC(bc.BaseClassifier):

    def __init__(self, classifier_name, dataset_name, perc_ds=1, perc_labeled=0.5, data_preparation=None, n_runs=5,
                 negative_classes=None, prefix_path=''):
        super().__init__(classifier_name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs,
                         negative_classes, prefix_path)

        self.batch_size_labeled = 256

    def get_sup_loss(self, n_elements, beta_same=1., beta_diff=1.):
        def my_sup_loss(y_true, y_pred):

            # calcolo coefficienti y
            y_same = tf.matmul(y_true, tf.transpose(y_true))
            y_diff = (y_same - 1.) * -1.

            # calcolo distanze
            r = tf.reduce_sum(y_pred * y_pred, 1)
            r = tf.reshape(r, [-1, 1])
            D = r - 2 * tf.matmul(y_pred, tf.transpose(y_pred)) + tf.transpose(r)
            distances = tf.linalg.band_part(D, 0, -1)

            final = 0

            # calcolo loss per quelli della stessa classe
            loss_same = tf.maximum(0., distances - beta_same)
            final += y_same * loss_same

            # calcolo loss per quelli di classe differente
            loss_diff = tf.maximum(0., beta_diff - distances)
            final += y_diff * loss_diff

            # viene presa la parte superiore della matrice quadrata
            final = tf.linalg.band_part(final, 0, 0)

            res = tf.reduce_sum(final)

            # normalizzazione in base al numero di elementi
            return res / ((n_elements ** 2 - n_elements) / 2)

        return my_sup_loss

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def get_model(self, input_dim, hyp):

        act = 'relu'
        init = 'glorot_uniform'
        # no weight regulizer

        # LAYERS
        dims = [input_dim, 500, 500, 2000, int(hyp["Embedding_dim"])]

        n_stacks = len(dims) - 1

        input_data = Input(shape=(dims[0],), name='input')
        x = input_data

        # internal layers of encoder
        for i in range(n_stacks - 1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i,)(x)

        # latent hidden layer
        encoded = Dense(dims[-1], activation='linear', kernel_initializer=init, name='encoder')(x)

        # internal layers of decoder
        x = encoded
        for i in range(n_stacks - 1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i,)(x)

        # decoder output
        x = Dense(dims[0], kernel_initializer=init, name='decoder',)(x)
        decoded = x

        return self.set_model_output(input_data, encoded, decoded, hyp)

    def set_model_output(self, input_data, encoded, decoded, hyp, include_clustering=False, centroids=None):

        gamma_kld = 0.1

        # supervised loss
        loss_labeled = ['mse', self.get_sup_loss(self.batch_size_labeled, hyp["Beta_sup"], hyp["Beta_sup"])]
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
        model_unlabeled = Model(inputs=input_data, outputs=output_unlabeled)
        model_labeled = Model(inputs=input_data, outputs=output_labeled)

        # compile models
        opt = tf.keras.optimizers.Adam()

        model_unlabeled.compile(loss=loss_unlabeled, loss_weights=loss_weights_unlabeled, optimizer=opt)
        model_labeled.compile(loss=loss_labeled, loss_weights=loss_weights_labeled, optimizer=opt)

        return model_unlabeled, model_labeled  # MIND THE ORDER

    def get_grid_hyperparameters(self):
        return {
            'Beta_sup': np.logspace(-1, 2, 3),
            'Gamma_sup': np.logspace(-1, 1, 3),
            'Embedding_dim': np.linspace(4, 16, 7)
        }

    def predict(self, model, x):
        # nearest centroid
        model[0].predict(x)[1].argmax(1)

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):

        model_unlabeled, model_labeled = model
        input_data = model_unlabeled.layers[0].input
        encoded = model_unlabeled.get_layer("encoder").output
        decoded = model_unlabeled.get_layer("decoder").output

        # first train
        #run_duplex()

        # kmeans for centroids
        centroids = self.get_centroids_from_kmeans(model_labeled, ds_unlabeled, ds_labeled, y_labeled)

        # models for the second step
        model_unlabeled, model_labeled = self.set_model_output(input_data, encoded, decoded, current_hyp, True, centroids)

        #run_duplex()
        # todo history

        return None

    def run_duplex(self, model_unlabeled, model_labeled, encoder,
               ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, ds_test, y_test,
               do_clustering, max_epochs):
        pass

    @tf.function
    def accuracy_metric(self, y_true, y_pred):
        return tf.numpy_function(func=self.func_accuracy, inp=[y_pred, y_true], Tout=[tf.float64])

    def func_accuracy(self, y_true, y_pred):
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
        print("Getting centroids from Kmeans...")

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
        #return K.eval(self.clusters)


