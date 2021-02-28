#copiato da https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/

from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.layers import InputSpec, Layer


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




# class ClusteringLayer(layers.Layer):
#     """
#     Clustering layer converts input sample (feature) to soft label.
#
#     # Example
#     ```
#         model.add(ClusteringLayer(n_clusters=10))
#     ```
#     # Arguments
#         n_clusters: number of clusters.
#         weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
#         alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
#     # Input shape
#         2D tensor with shape: `(n_samples, n_features)`.
#     # Output shape
#         2D tensor with shape: `(n_samples, n_clusters)`.
#     """
#
#     def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#         super(ClusteringLayer, self).__init__(**kwargs)
#         self.n_clusters = n_clusters
#         self.alpha = alpha
#         self.initial_weights = weights
#         self.input_spec = layers.InputSpec(ndim=2)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 2
#         input_dim = input_shape[1]
#         self.input_spec = layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
#         self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#         self.built = True
#
#     def get_centroids(self):
#         return self.clusters.numpy()
#
#     def call(self, inputs, **kwargs):
#         """ student t-distribution, as same as used in t-SNE algorithm.
#                  q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
#                  q_ij can be interpreted as the probability of assigning sample i to cluster j.
#                  (i.e., a soft assignment)
#         Arguments:
#             inputs: the variable containing data, shape=(n_samples, n_features)
#         Return:
#             q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
#         """
#         q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
#         q **= (self.alpha + 1.0) / 2.0
#         q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
#         return q
#
#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) == 2
#         return input_shape[0], self.n_clusters
#
#     def get_config(self):
#         config = {'n_clusters': self.n_clusters}
#         base_config = super(ClusteringLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))




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


def get_my_argmax_loss(n_elements = 256):
    def my_argmax_loss(y_true, y_pred):

        # n_elements = y_pred.shape[0]
        #n_elements = 256 #dovrebbe venire calcolato
        n_values = y_pred.shape[1]


        y = tf.reshape(tf.tile(y_true, tf.constant([n_elements, 1])), (n_elements, n_elements, n_values))
        y_i = tf.reshape(tf.tile(y_true, tf.constant([1, n_elements])), (n_elements, n_elements, n_values))

        y_prod = tf.reduce_sum(tf.multiply(y, y_i), 2) * 2 - 1

        m = tf.reshape(tf.tile(y_pred, tf.constant([n_elements, 1])), (n_elements, n_elements, n_values))
        m_i = tf.reshape(tf.tile(y_pred, tf.constant([1, n_elements])), (n_elements, n_elements, n_values))

        m_i = tf.math.log(m_i) * -1

        m_prod = tf.reduce_sum(tf.multiply(m, m_i), 2)

        final = m_prod * y_prod

        final = tf.linalg.set_diag(final, tf.zeros(n_elements))

        res = tf.reduce_sum(final, axis=0)

        return res

    return my_argmax_loss