#copiato da https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/

from tensorflow.keras import layers
import tensorflow.keras.backend as K


class ClusteringLayer(layers.Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_centroids(self):
        return self.clusters.numpy()

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



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
        diff = expand - self.centroid
        sq =K.square(diff)
        sum = K.sum(sq, axis=2)

        #sq1 = K.square(self.radius) #credo sia errato inserire il raggio al quadrato
        sq1 = self.radius

        q = sum - sq1

        res =  1.0 - q #hinge loss trick (q must be 0 or negative)

        return res

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return 1, 1

    def get_config(self):
        config = {'centroid': self.centroid, 'radius': radius}
        base_config = super(OneClassLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
