#copiato da https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/

from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.layers import InputSpec, Layer
import numpy as np
import get_data
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment as linear_assignment
#from sklearn.utils.linear_assignment_ import linear_assignment


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


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


class OneClassLayer(layers.Layer):

    def __init__(self, initial_centroid, initial_radius, weights=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(OneClassLayer, self).__init__(**kwargs)

        self.initial_centroid = initial_centroid
        self.initial_radius = initial_radius

        self.initial_weights = weights
        self.input_spec = layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.centroid = self.add_weight(shape=(1, input_dim), initializer='glorot_uniform', name='centroid')
        self.radius = self.add_weight(shape=(1, 1), initializer='glorot_uniform', name='radius')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_centroid(self):
        return self.centroid

    def get_radius(self):
        return self.radius

    def call(self, inputs, **kwargs):

        self.add_loss(K.square(self.radius))

        expand = K.expand_dims(inputs, axis=1)
        sq = K.square(expand - self.centroid)
        sum = K.sqrt(K.sum(sq, axis=2))

        #sq1 = K.square(self.radius) #credo sia errato inserire il raggio al quadrato
        sq1 = self.radius

        res =  1.0 - (sum - sq1) #hinge loss trick (q must be 0 or negative)
        return res

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return 1, 1

    def get_config(self):
        config = {'centroid': self.centroid, 'radius': self.radius}
        base_config = super(OneClassLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def get_my_hinge_loss(n_samples, v=1):
    def my_hinge_loss(y_true, y_pred):
        return tf.keras.losses.hinge(y_true, y_pred) / (n_samples * v)

    return my_hinge_loss



class MultiClassLayer(layers.Layer):

    def __init__(self, initial_centroids, initial_radiuses, weights=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MultiClassLayer, self).__init__(**kwargs)

        self.initial_centroids = initial_centroids
        self.initial_radiuses = initial_radiuses

        self.initial_weights = weights
        self.input_spec = layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.centroids = self.add_weight(shape=(len(self.initial_centroids), input_dim), initializer='glorot_uniform', name='centroid')
        self.radiuses = self.add_weight(shape=(len(self.initial_radiuses), 1), initializer='glorot_uniform', name='radius')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_centroids(self):
        return self.centroids

    def get_radiuses(self):
        return self.radiuses

    def call(self, inputs, **kwargs):

        self.add_loss(K.square(self.radiuses))

        expand = K.expand_dims(inputs, axis=1)
        sq = K.square(expand - self.centroid) #bisogna restituire il minimo
        sum = K.sqrt(K.sum(sq, axis=2))

        #sq1 = K.square(self.radius) #credo sia errato inserire il raggio al quadrato
        sq1 = self.radius

        res =  1.0 - (sum - sq1) #hinge loss trick (q must be 0 or negative)
        return res

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return 1, 1

    def get_config(self):
        config = {'centroids': self.centroids, 'radiuses': self.radiuses}
        base_config = super(MultiClassLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





class ArgMaxClusterLayer(Layer):
    def __init__(self, weights=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ArgMaxClusterLayer, self).__init__(**kwargs)

        self.initial_weights = weights
        #self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        #input_dim = input_shape[1]
        #self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):

        res =  inputs.argmax(1)
        return res

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], 1

    def get_config(self):
        config = {}
        base_config = super(ArgMaxClusterLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_my_argmax_loss(n_elements=256, y_prod_type='all', m_prod_type="diff"):
    def my_argmax_loss(y_true, y_pred):

        #n_elements = y_pred.shape[0] #dovrebbe venire calcolato
        n_values = y_pred.shape[1]

        # calcolo coefficienti y
        y = tf.reshape(tf.tile(y_true, tf.constant([n_elements, 1])), (n_elements, n_elements, n_values))
        y_i = tf.reshape(tf.tile(y_true, tf.constant([1, n_elements])), (n_elements, n_elements, n_values))

        if y_prod_type == "same":
            y_prod = tf.reduce_sum(tf.multiply(y, y_i), 2) # +1 e 0 (solo uguali)
        elif y_prod_type == "diff":
            y_prod = tf.reduce_sum(tf.multiply(y, y_i), 2) - 1 # 0 e -1 (solo diversi)
        else:
            y_prod = tf.reduce_sum(tf.multiply(y, y_i), 2) * 2 - 1  # +1 e -1  (uguali e diversi)


        m = tf.reshape(tf.tile(y_pred, tf.constant([n_elements, 1])), (n_elements, n_elements, n_values))
        m_i = tf.reshape(tf.tile(y_pred, tf.constant([1, n_elements])), (n_elements, n_elements, n_values))

        # calcolo cross entropy
        if m_prod_type == "ce":
            m_i = tf.math.log(m_i) * -1
            m_prod = tf.reduce_sum(tf.multiply(m, m_i), 2)
        # calcolo somiglianza
        elif m_prod_type == "diff":
            m_prod = tf.reduce_sum(tf.abs(tf.subtract(m, m_i)), 2)
        # calcolo moltiplicazione
        elif m_prod_type == "molt":
            m_prod = tf.reduce_sum(tf.multiply(m, m_i), 2) * -1

        final = m_prod * y_prod

        final = tf.linalg.set_diag(final, tf.zeros(n_elements)) # gli elementi diagonali vengono rimossi

        res = tf.reduce_sum(final, axis=0)

        return res

    return my_argmax_loss


def compute_centroids_from_labeled(encoder, x, y, positive_classes):
    # calcolo come media dei valori della stessa classe
    centroids = []

    x_labeled_encoded = encoder.predict(x)
    for y_class in positive_classes:
        only_x_class, _ = get_data.filter_ds(x_labeled_encoded, y, [y_class])
        centroids.append(np.mean(only_x_class, axis=0))

    return np.array(centroids)


def get_centroids_from_kmeans(num_classes, positive_classes, x_unlabeled, x_labeled, y, encoder, init_kmeans=True, centroids = []):

    all_x_encoded = encoder.predict(np.concatenate((x_labeled, x_unlabeled), axis=0))

    if init_kmeans:

        if len(centroids) == 0:
            centroids = compute_centroids_from_labeled(encoder, x_labeled, y, positive_classes)

        center = np.mean(centroids, axis=0)
        radius_from_center = np.max([np.abs(center - centroid) for centroid in centroids])

        best_kmeans = None

        for i in range(num_classes * 40):

            # si aggiungono dei centroidi per le classi negative. Esse sono centrate e hanno una scala di riferimento
            try_centroids = centroids.copy()
            while len(try_centroids) < num_classes:
                new_c = np.random.normal(center, radius_from_center)
                try_centroids = np.concatenate((try_centroids, [new_c]), axis=0)
            try_centroids = np.array(try_centroids)

            kmeans = KMeans(n_clusters=num_classes, init=try_centroids, n_init=1)
            kmeans.fit(all_x_encoded)

            if best_kmeans is None or kmeans.inertia_ < best_kmeans.inertia_:
                best_kmeans = kmeans

    else:
        # senza inizializzazione
        best_kmeans = KMeans(n_clusters=num_classes, n_init=num_classes * 40)

        best_kmeans.fit(all_x_encoded)

    return best_kmeans.cluster_centers_


def print_measures(y_true, y_pred, classes, ite=None, x_for_silouhette=None):

    # calcolo dell'indice di purezza
    purity = 0
    for index in range(len(classes)):
        # si ottengono gli esempi del cluster i-esimo
        mask = [y == index for y in y_pred]
        cluster_i = [y for i, y in enumerate(y_true) if mask[i]]

        # si ottiene la classe che occorre di piu nel cluster
        max_y_class = 0
        for c in classes:
            y_class = sum([1 for y in cluster_i if y == c])
            if y_class > max_y_class:
                max_y_class = y_class

        purity += max_y_class
    purity = purity / len(y_true)

    # purezza rispetto alle classi piuttosto che ai cluster
    purity_class = 0
    for index in range(len(classes)):
        # si ottengono gli esempi della classe i-esima
        mask = [y == index for y in y_true]
        class_i = [y for i, y in enumerate(y_pred) if mask[i]]

        # si ottiene il cluster che occorre di piu per la classe
        max_y_class = 0
        for c in classes:
            y_class = sum([1 for y in class_i if y == c])
            if y_class > max_y_class:
                max_y_class = y_class

        purity_class += max_y_class
    purity_class = purity_class / len(y_true)

    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
    fwm = fowlkes_mallows_score(y_true, y_pred)

    # cluster accuracy
    y_true1 = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true1.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true1[i]] += 1

    row, col = linear_assignment(w.max() - w)
    #acc = sum([w[i, j] for i, j in (row, col)]) * 1.0 / y_pred.size
    acc = sum([a for a in w[row, col]]) * 1.0 / y_pred.size

    format = "{:5.3f}"
    print("Ite:", "{:4.0f}".format(ite) if ite is not None else "-", "- Purity:", format.format(purity),
          "- NMI:", format.format(nmi), "- ARI:",  format.format(ari), "- FOW:",  format.format(fwm),
          "- Purity class:",  format.format(purity_class), "- Acc:", acc)

    if x_for_silouhette is not None:
        sil = silhouette_score(x_for_silouhette, y_pred, metric='euclidean')
        print("Silhouette:",  format.format(sil))


