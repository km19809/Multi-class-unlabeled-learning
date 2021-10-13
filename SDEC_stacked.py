import base_classifier as bc
import numpy as np
import keras
from keras import Model, Input
from keras.layers import Dense, Layer, InputSpec, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.cluster import KMeans
import datasets as ds
import tensorflow.keras.backend as K
import gc
import math
from keras.callbacks import History


# Supervised deep embedded clustering
# This is a novel approach in a semi-supervised setting that combines the concepts:
# autoencoding, clustering and metric learning
class SDECStacked(bc.BaseClassifier):

    @staticmethod
    def get_sup_loss(beta_same=1., beta_diff=1.):
        '''Returns the loss function for the metric learning component'''
        def my_sup_loss(y_true, y_pred):

            # computing coefficients based on the class labels
            y_same = tf.matmul(y_true, tf.transpose(y_true))
            y_diff = (y_same - 1.) * -1.

            # computing distances in the embedded space
            r = tf.reduce_sum(y_pred * y_pred, 1)
            r = tf.reshape(r, [-1, 1])
            distances = r - 2 * tf.matmul(y_pred, tf.transpose(y_pred)) + tf.transpose(r)

            # loss for instances of the same class
            final = y_same * tf.maximum(0., distances - beta_same)

            # loss for instances of different classes
            final += y_diff * tf.maximum(0., beta_diff - distances)

            # just taken the upper part of the square matrix (the distance matrix is symmetric)
            final = tf.linalg.band_part(final, 0, -1)
            final = tf.reduce_sum(final)

            n_elements = tf.cast(tf.shape(y_pred)[0], 'float32')  # number of samples

            # normalization based on the number of the samples
            return final / (1 + (n_elements ** 2 - n_elements) / 2)

        return my_sup_loss

    @staticmethod
    def target_distribution(q):
        ''' This method computes the target distribution called P '''
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def get_model(self, input_dim, hyp):

        act = 'relu'
        init = 'glorot_uniform'

        # fixed weight regularizer
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

        # latent hidden layer (linear activation)
        encoded = Dense(dims[-1], activation='linear', kernel_initializer=init, name='encoder_%d' % (n_stacks - 1),
                        kernel_regularizer=keras.regularizers.l2(w_dec))(x)

        # internal layers of decoder
        x = encoded
        for i in range(n_stacks - 1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i,
                      kernel_regularizer=keras.regularizers.l2(w_dec))(x)

        # decoder output (linear activation)
        x = Dense(dims[0], kernel_initializer=init, name='decoder_0',)(x)
        decoded = x

        return self.set_model_output(input_data, encoded, decoded, hyp)

    def get_stacked_model(self, input_dim, output_dim, first_pair, last_pair):

        init = 'glorot_uniform'

        input_data = Input(shape=(input_dim,), name='input')
        x = input_data
        x = Dropout(0.2)(x)
        x = Dense(output_dim, name='encoder', activation='relu' if not last_pair else 'linear', kernel_initializer=init)(x)

        x = Dropout(0.2)(x)
        output = Dense(input_dim, name='decoder', activation='relu' if not first_pair else 'linear', kernel_initializer=init)(x)

        model_unlabeled = Model(inputs=input_data, outputs=output)
        model_unlabeled.compile(loss='mse', optimizer=Adam())

        return model_unlabeled

    def set_model_output(self, input_data, encoded, decoded, hyp, include_clustering=False, centroids=None):
        '''Returns the models and constructs their outputs'''

        # for some experiments the reconstruction loss (as well other losses) is set to zero
        gamma_rec = 0. if (self.ablation_type == 2 and centroids is None) or (self.ablation_type == 4 and centroids is not None) else 1.

        # labeled model output
        gamma_supervised = 0. if (self.ablation_type == 1 and centroids is None) or (self.ablation_type == 5 and centroids is not None) else hyp['Gamma_sup']

        # fixed gamma for the clustering loss component
        gamma_kld = 0. if self.ablation_type == 3 else 0.1

        loss_labeled = ['mse', self.get_sup_loss(hyp["Beta_sup"], hyp["Beta_sup"])]
        loss_weights_labeled = [gamma_rec, gamma_supervised]

        output_labeled = [decoded, encoded]

        # unlabeled model output
        loss_unlabeled = ['mse']
        output_unlabeled = [decoded]
        loss_weights_unlabeled = [gamma_rec]

        if include_clustering:

            # create extra layer with the given centroids
            unlabeled_last_layer = ClusteringLayer(len(self.classes), weights=[centroids], name='clustering')(encoded)

            # add to the models a new output and a new loss component
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
        # fixed learning parameter and weight decay
        # Beta_sup_same: margin for instances of the same class
        # Beta_sup_diff: margin for instances of different classes

        if self.validate_hyp:
            return {
                'Beta_sup': np.logspace(0, 2, 3),
                'Gamma_sup': np.logspace(-2, -1, 2),
            }
        else:
            return {
                'Beta_sup': [10],
                'Gamma_sup': [0.1],
            }

    def predict(self, model, x):
        model_unlabeled = model[0]

        # distance from each centroid
        pred = model_unlabeled.predict(x)

        # the class with nearest centroid is taken
        return pred[1].argmax(1)

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):

        def run_pretraining(model_unlabeled, model_labeled, ds_labeled,
                            y_labeled_original, ds_unlabeled, epochs_pretraining):
            '''This method trains the model using the reconstruction loss and the metric learning loss
            No clustering is implemeted here'''

            # history of the loss
            history = dict()
            history["loss_rec"] = []
            history["loss_sup"] = []

            y_labeled = tf.keras.utils.to_categorical(y_labeled_original, len(self.classes))  # categorical labels

            # batch sizes for labeled and unlabeled instances
            bs_unlab = 256
            bs_lab = 256
            epoch = 0

            # run for a given number of epochs
            while epoch < epochs_pretraining:
                # print("EPOCH {}".format(epoch))

                if epoch % 50 == 0:
                    gc.collect()

                # shuffle labeled and unlabeled instance sets
                shuffler_l = np.random.permutation(len(ds_labeled))
                ds_labeled = ds_labeled[shuffler_l]
                y_labeled = y_labeled[shuffler_l]

                shuffler_l = np.random.permutation(len(ds_unlabeled))
                ds_unlabeled = ds_unlabeled[shuffler_l]

                # epoch variables
                i_unlab = 0
                i_lab = 0
                epoch_losses = []  # loss of each batch of samples
                finish_labeled = False
                finish_unlabeled = False

                while not (finish_labeled and finish_unlabeled):

                    # unlabeled mini-batch
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

                    # labeled mini-batch
                    if not finish_labeled:
                        if (i_lab + 1) * bs_lab >= ds_labeled.shape[0]:

                            t_labeled = [ds_labeled[i_lab * bs_lab::], y_labeled[i_lab * bs_lab::]]
                            b_labeled = ds_labeled[i_lab * bs_lab::]

                            finish_labeled = True
                        else:
                            t_labeled = [ds_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab],
                                         y_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab]]
                            b_labeled = ds_labeled[i_lab * bs_lab:(i_lab + 1) * bs_lab]

                            i_lab += 1

                        losses = model_labeled.train_on_batch(b_labeled, t_labeled)
                        epoch_losses.append(losses[1:])

                # computing loss for each epoch
                losses = np.sum(epoch_losses, axis=0)
                history["loss_rec"].append(losses[0])
                history["loss_sup"].append(losses[1])

                epoch += 1

            return history

        def run_clustering(model_unlabeled, model_labeled, ds_labeled_original, y_labeled_original,
                           ds_unlabeled_original, y_unlabeled,
                           x_test, y_test, maxiter):
            '''This methods trains the model with all the three loss components'''

            # History for the loss
            history = dict()
            history["loss_rec"] = []
            history["loss_sup"] = []
            history["loss_clu"] = []
            history["accuracy_metric"] = []
            history["val_accuracy_metric"] = []

            # tolerance threshold for the convergence criterium
            tol = 0.001

            # batch size for labeled and unlabeled instances
            bs_unlab = 256
            bs_lab = 256

            # some info from all the dataset
            all_x = np.concatenate((ds_labeled_original, ds_unlabeled_original), axis=0)
            all_y = np.concatenate((y_labeled_original, y_unlabeled), axis=0) # used only for the accuracy metric

            # index of labeled/unlabeled samples for the dataset
            labeled_indexes = np.array([i < len(ds_labeled_original) for i, _ in enumerate(all_x)])
            unlabeled_indexes = np.array([i >= len(ds_labeled_original) for i, _ in enumerate(all_x)])

            y_labeled_cat_original = tf.keras.utils.to_categorical(y_labeled_original, len(self.classes)) # categorical labels

            y_pred_last = None  # last predictions for the instances (used for the convergence criterium)
            p = None  # target distribution P
            stop_for_delta = False  # convergence criterium
            batch_n = 0  # mini-batch number
            num_before_update = 0  # number of batchs before updating the target probability
            epoch = 0

            plot_interval = 200  # interval of epochs in order to make a new plot for the clusters
            clustering_data_plot = dict()  # data for the plots

            # the iteration can stop for a convergence criterium or for a maximum number of iterations
            while batch_n < maxiter and not stop_for_delta:

                if epoch % plot_interval == 0:
                    # cluster plot data (no info stored)
                    clustering_data_plot[epoch] = {
                        # 'centroids': model_unlabeled.get_layer('clustering').get_centroids(),
                        # 'x_data': model_labeled.predict(all_x)[1],
                        # 'y_data': all_y,
                        # 'y_pred': self.predict((model_unlabeled,), all_x),
                        # 'lab_index': labeled_indexes
                    }

                # print("EPOCH {}, Batch n° {}".format(epoch, batch_n))
                if epoch % 50 == 0:
                    gc.collect()

                # shuffle labeled dataset
                shuffler_l = np.random.permutation(len(ds_labeled_original))
                ds_labeled = ds_labeled_original[shuffler_l]
                y_labeled = y_labeled_cat_original[shuffler_l]

                # shuffle unlabeled dataset
                shuffler_u = np.random.permutation(len(ds_unlabeled_original))
                ds_unlabeled = ds_unlabeled_original[shuffler_u]

                if p is not None:
                    p_lab = p[labeled_indexes][shuffler_l]
                    p_unlab = p[unlabeled_indexes][shuffler_u]

                # epoch variables
                i_unlab = 0
                i_lab = 0
                ite = 0
                epoch_losses = [[0., 0., 0.]]

                finish_labeled = False
                finish_unlabeled = False

                while not (finish_labeled and finish_unlabeled):

                    # update target probability if the update interval is reached
                    if num_before_update <= 0:

                        num_before_update = self.update_interval  # reset the counter

                        # predict cluster probabilities
                        q = model_unlabeled.predict(all_x)[1]

                        # check convergence criterion
                        y_pred_new = q.argmax(1)
                        if y_pred_last is not None:
                            # get the number of changed labels
                            delta_label = sum(y_pred_new[i] != y_pred_last[i] for i in range(len(y_pred_new))) / y_pred_new.shape[0]

                            if delta_label < tol:
                                print('  Reached stopping criterium, delta_label ', delta_label, '< tol ', tol, '. Iter n°', batch_n)
                                stop_for_delta = True
                                break

                        # store predicted labels
                        y_pred_last = y_pred_new

                        # update the target distribution p and split it for the labeled and unlabeled instance sets
                        p = self.target_distribution(q)
                        p_lab = p[labeled_indexes][shuffler_l]
                        p_unlab = p[unlabeled_indexes][shuffler_u]

                    # unlabeled mini-batch
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
                        num_before_update -= 1

                    # labeled mini-batch
                    if not finish_labeled:
                        if (i_lab + 1) * bs_lab >= ds_labeled.shape[0]:

                            t_labeled = [ds_labeled[i_lab * bs_lab::], y_labeled[i_lab * bs_lab::],
                                         p_lab[i_lab * bs_lab::]]
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
                        num_before_update -= 1

                    ite += 1

                # computing loss for each epoch
                losses = np.sum(epoch_losses, axis=0)
                history["loss_rec"].append(losses[0])
                history["loss_sup"].append(losses[1])
                history["loss_clu"].append(losses[2])

                # accuracies on training and test sets
                history["accuracy_metric"].append(self.get_accuracy(self.predict((model_unlabeled,), all_x), all_y))
                history["val_accuracy_metric"].append(self.get_accuracy(self.predict((model_unlabeled,), x_test), y_test))

                epoch += 1

            del p, q, p_lab, p_unlab, all_x, all_y, ds_labeled, ds_unlabeled

            # cluster data (no variables stored due to memory usage overflow)
            clustering_data_plot[epoch] = {
                # 'centroids': model_unlabeled.get_layer('clustering').get_centroids(),
                # 'x_data': model_labeled.predict(all_x)[1],
                # 'y_data': all_y,
                # 'y_pred': self.predict((model_unlabeled,), all_x),
                # 'lab_index': labeled_indexes
            }

            return history, epoch, clustering_data_plot

        # number of epochs for the pre training step
        epochs_pretraining = 100

        # max iterations for the clustering step
        max_iter_clustering = 10000

        # getting some model variables
        model_unlabeled, model_labeled = model
        input_data = model_unlabeled.layers[0].input
        encoded = model_unlabeled.get_layer("encoder_3").output
        decoded = model_unlabeled.get_layer("decoder_0").output

        # setting initial weights for autoencoder
        models_stacked = self.data_for_run['models_stacked']
        for i in range(len(models_stacked)):
            model_stacked = models_stacked[i]
            model_labeled.get_layer('encoder_%d' % i).set_weights(model_stacked.get_layer('encoder').get_weights())
            model_unlabeled.get_layer('encoder_%d' % i).set_weights(model_stacked.get_layer('encoder').get_weights())

            model_labeled.get_layer('decoder_%d' % i).set_weights(model_stacked.get_layer('decoder').get_weights())
            model_unlabeled.get_layer('decoder_%d' % i).set_weights(model_stacked.get_layer('decoder').get_weights())

        # pre-training
        history_pre = run_pretraining(model_unlabeled, model_labeled, ds_labeled, y_labeled, ds_unlabeled, epochs_pretraining)

        # getting centroids with the custom kmeans algorithm
        centroids = self.get_centroids_from_kmeans(model_labeled, ds_unlabeled, ds_labeled, y_labeled)

        # setting output models for the clustering step
        model_unlabeled, model_labeled = self.set_model_output(input_data, encoded, decoded, current_hyp, True, centroids)
        model = (model_unlabeled, model_labeled)

        # clustering step
        history_clu, epochs_clu, data_plot = run_clustering(model_unlabeled, model_labeled,
                                                                 ds_labeled, y_labeled, ds_unlabeled, y_unlabeled,
                                                                 x_test, y_test, max_iter_clustering)

        # set accuracy history object (used for plotting)
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
        # history.data_plot = data_plot  # too much data to store (memory overflow)

        return model, history

    def run_preparation(self, ds_labeled, y_labeled, ds_unlabeled):

        # define stacked models (no hyperparameters needed)
        ds_all = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
        models_stacked = []

        dims = [ds_labeled[0].shape[0], 500, 500, 2000, 10]
        epochs_stacked = 150

        for i in range(len(dims) - 1):
            model_stacked = self.get_stacked_model(dims[i], dims[i + 1], i == 0, i == len(dims) - 1)

            model_stacked.fit(ds_all, ds_all, batch_size=256, epochs=epochs_stacked, shuffle=True, verbose=0)
            models_stacked.append(model_stacked)

            model_for_new_input = Model(model_stacked.input, model_stacked.get_layer('encoder').output)
            ds_all = model_for_new_input.predict(ds_all)

        self.data_for_run['models_stacked'] = models_stacked

    def accuracy_metric(self, y_true, y_pred):
        raise Exception("Not implemented")
        # return tf.py_function(func=self.get_accuracy, inp=[y_pred, y_true], Tout=[tf.float32])

    # commented, using the base method
    '''@staticmethod
    def get_accuracy(y_pred, y_true):
        # cluster accuracy computed with the linear assignment
        y_true1 = y_true.astype(np.int64)

        D = max(y_pred.max(), y_true1.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)

        for i in range(y_pred.size):
            w[y_pred[i], y_true1[i]] += 1

        row, col = linear_assignment(w.max() - w)

        acc = sum([a for a in w[row, col]]) * 1.0 / y_pred.size

        return acc'''

    def get_centroids_from_kmeans(self, model_labeled, x_unlabeled, x_labeled, y):

        def get_initial_centroids(X, k, centers=None):
            '''This method returns initial centroids from kmeans'''

            def dist(data, centers):
                # euclidean distance of each point with each center
                return np.sum((np.array(centers) - data[:, None, :]) ** 2, axis=2)

            # Sample the first point if not already defined
            if centers is None or len(centers) == 0:
                initial_index = np.random.choice(range(X.shape[0]), )
                centers = np.array([X[initial_index, :].tolist()])

            # choose K centers
            while len(centers) < k:
                distance = dist(X, np.array(centers))

                # if len(centers) == 0:
                #    pdf = distance / np.sum(distance)
                #    centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf.flatten())]
                # else:
                # Calculate the distance of each point from its nearest centroid
                dist_min = np.min(distance, axis=1)

                pdf = dist_min / np.sum(dist_min)  # probability density function

                if any([math.isnan(x) for x in pdf]):
                    # if there is some floating point problem, just take the farthest point
                    index_max = np.argmax(dist_min, axis=0)
                    centroid_new = X[index_max, :]
                else:
                    # Sample one point from the distribution given from the distance array
                    centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf)]

                centers = np.concatenate((centers, [centroid_new.tolist()]), axis=0)

            return np.array(centers)

        # merging all instances
        if len(x_labeled) > 0:
            all_ds = np.concatenate((x_labeled, x_unlabeled), axis=0)
        else:
            all_ds = x_unlabeled

        all_x_encoded = model_labeled.predict(all_ds)[1]  # get points in the embedded space

        # getting initial centroids for positive classes
        centroids = []
        if len(x_labeled) > 0:
            x_labeled_encoded = model_labeled.predict(x_labeled)[1]  # encoded points

            for y_class in self.positive_classes:
                # for each class, the centroid is computed as the mean value of the instances of the given class
                only_x_class, _ = ds.filter_ds(x_labeled_encoded, y, [y_class])
                if len(only_x_class) > 0:
                    centroids.append(np.mean(only_x_class, axis=0))

        centroids = np.array(centroids)

        # multiple execution for k-means
        best_kmeans = None

        for i in range(len(self.classes)):

            # getting centroids with the k-means++ method
            initial_centroids = get_initial_centroids(all_x_encoded, len(self.classes), centroids)

            # run k-means
            kmeans = KMeans(n_clusters=len(self.classes), init=initial_centroids, n_init=1)
            kmeans.fit(all_x_encoded)

            # just maintain the model with the lowest inertia value
            if best_kmeans is None or kmeans.inertia_ < best_kmeans.inertia_:
                best_kmeans = kmeans

        del all_ds

        # returning centroids
        return best_kmeans.cluster_centers_


# Custom layer for the clustering component of SDEC
# This layer takes the centroids as input and then computes the soft assignments for each class
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

        # the centroids are a variable to be learnt
        self.centroids = self.add_weight(name='centroids', shape=(self.n_clusters, input_dim), initializer='glorot_uniform')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        # compute the soft assignments
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.centroids), axis=2) / self.alpha))
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
        # returns centroids array
        return self.centroids.numpy()


