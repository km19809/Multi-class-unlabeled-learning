import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.datasets import mnist
from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import custom_layers, get_data
import keras
from sklearn.metrics import silhouette_score
from sklearn import metrics

from scipy.optimize import linear_sum_assignment as linear_assignment
#from sklearn.utils.linear_assignment_ import linear_assignment

from keras.utils import plot_model
from IPython.display import Image


classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def create_autoencoder(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1

    input_data = Input(shape=(dims[0],), name='input')
    x = input_data

    # internal layers of encoder
    for i in range(n_stacks - 1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
    # latent hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)
    x = encoded
    # internal layers of decoder
    for i in range(n_stacks - 1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)
    # decoder output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)

    decoded = x

    autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')
    encoder_model = Model(inputs=input_data, outputs=encoded, name='encoder')

    return autoencoder_model, encoder_model


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)

    x, y = get_data.filter_ds(x, y, classes)

    # 10 clusters
    n_clusters = len(np.unique(y))

    # autoencoder
    n_epochs = 50
    batch_size = 256
    dims = [x.shape[-1], 500, 500, 2000, 10]

    autoencoder, encoder = create_autoencoder(dims)
    #autoencoder.compile(optimizer=SGD(lr=1, momentum=0.9), loss='mse')
    autoencoder.compile(optimizer=Adam(), loss='mse')

    # TRAINING (se i parametri sono stati già salvati, li si prende da file system)
    model_loaded = False
    name_file_model = 'parameters/test_only_clustering'

    try:
        autoencoder.load_weights(name_file_model)
        model_loaded = True
    except Exception:
        pass

    if not model_loaded:
        autoencoder.fit(x, x, batch_size=batch_size, epochs=n_epochs)
        autoencoder.save_weights(name_file_model)


    # clustering
    maxiter = 10000
    update_interval = 140
    tol = 0.001  # tolerance threshold to stop training
    index_array = np.arange(x.shape[0])

    '''
    clustering_layer = custom_layers.ClusteringLayer(n_clusters, name="clustering")
    cl = clustering_layer(encoder.output)
    model = Model(inputs=encoder.input, outputs=cl)

    # Initialize cluster centers using k-means.
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    model_loaded = False
    name_file_model2 = 'parameters/test_only_clustering_2'

    try:
        model.load_weights(name_file_model2)
        model_loaded = True
    except Exception:
        pass

    index = 0
    maxiter = 10000
    update_interval = 100
    tol = 0.001  # tolerance threshold to stop training
    index_array = np.arange(x.shape[0])
    if not model_loaded:
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = model.predict(x, verbose=0)
                p = custom_layers.target_distribution(q)

            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            print("Current loss is: " + str(loss))

        model.save_weights(name_file_model2)

    # accuratezza e altri indici
    #centroids = clustering_layer.get_centroids()
    y_pred = print_accuracy(x, y, centroids, "first acuracy DEC", model, encoder)
    x_embedded_encoder = encoder.predict(x)

    #score = silhouette_score(x_embedded_encoder, y_pred, metric='euclidean')
    #print("Silouhette score:" + str(score))
    #plot_2d(x_embedded_encoder, y_pred, centroids)
'''




    # DEC
    #autoencoder, encoder = create_autoencoder(dims)
    #autoencoder.load_weights(name_file_model)
    clustering_layer = custom_layers.ClusteringLayer(n_clusters, name='clustering')
    cl = clustering_layer(encoder.output)
    model = Model(inputs=encoder.input, outputs=[cl, autoencoder.output])

    plot_model(model, to_file='model.png', show_shapes=True)
    Image(filename='model.png')

    # run k means for cluster centers
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    y_pred_last = np.copy(y_pred)

    run_inter(model, x, y, y_pred_last, index_array, batch_size, maxiter, tol, update_interval, 0.1, 1)

    # accuratezza e altri indici
    centroids = clustering_layer.get_centroids()
    y_pred = print_accuracy(x, y, centroids, "after DEC", model, encoder, second=True)

    x_embedded_encoder = encoder.predict(x)
    score = silhouette_score(x_embedded_encoder, y_pred, metric='euclidean')
    print("Silouhette score:" + str(score))
    plot_2d(x_embedded_encoder, y_pred, centroids)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in enumerate(ind)]) * 1.0 / y_pred.size


def run_inter(model, x, y, y_pred_last, index_array, batch_size, maxiter, tol, update_interval, kld_weight, mse_weight):

    # model.compile(loss=['kld', 'mse'], loss_weights=[kld_weight, mse_weight], optimizer=SGD(0.01, 0.9))
    model.compile(loss=['kld', 'mse'], loss_weights=[kld_weight, mse_weight], optimizer=Adam())

    loss = [0, 0, 0]
    index = 0
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q, _ = model.predict(x, verbose=0)
            p = custom_layers.target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            if y is not None:
                acc = np.round(cluster_acc(y, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                loss = np.round(loss, 5)

                print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

            # check stop criterion
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                'Reached tolerance threshold. Stopping training.'
                break

        # train on batch
        if (index + 1) * batch_size > x.shape[0]:
            loss = model.train_on_batch(x=x[index * batch_size::],
                                             y=[p[index * batch_size::], x[index * batch_size::]])
            index = 0
        else:
            loss = model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                             y=[p[index * batch_size:(index + 1) * batch_size],
                                                x[index * batch_size:(index + 1) * batch_size]])
            index += 1

        #ite += 1


    '''index = 0
    loss = -1

    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q, _ = model.predict(x, verbose=0)
            p = custom_layers.target_distribution(q)
            y_pred = q.argmax(1)
            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)

            print("delta_labels is: " + str(delta_label))
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
        loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        print("Current loss is: " + str(loss))'''


def print_accuracy(x, y, centroids, label, model, encoder, second=False):

    mapping_indexes = dict()

    for y_class in classes:

        only_x_class, _ = get_data.filter_ds(x, y, [y_class])
        only_x_class = encoder.predict(only_x_class)

        centroid_class = np.mean(only_x_class, axis=0)

        index_nearest_centroid = np.argmin([np.linalg.norm(centroid - centroid_class) for centroid in centroids])

        mapping_indexes[index_nearest_centroid] = y_class

    q = model.predict(x, verbose=0)

    if second:
        q = q[0]
    y_pred = q.argmax(1)

    # si ottengono i valori delle classi
    y_pred = [mapping_indexes[i] if i in mapping_indexes else -1 for i in y_pred]

    acc = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            acc += 1

    acc = acc * 100 / len(y)

    print("Accuracy for " + label + ":" + str(acc))

    return y_pred



def plot_2d(x_embedded_encoder, y_pred, centroids):
    from sklearn.manifold import TSNE
    x_embedded = TSNE(n_components=2, verbose=1).fit_transform(np.concatenate((x_embedded_encoder, centroids), axis=0))
    vis_x = x_embedded[:-len(centroids), 0]
    vis_y = x_embedded[:-len(centroids), 1]
    plt.scatter(vis_x, vis_y, c=y_pred, cmap=plt.cm.get_cmap("jet", 256), linewidths=0.2, marker=".")
    plt.colorbar(ticks=range(256))
    plt.clim(-0.5, 9.5)

    # scatter centroids
    COLORS = np.array([
        '#FF3333',  # red
        '#FF7216',  # orange
        '#FCD116',  # yellow
        '#0198E1',  # blue
        '#BF5FFF',  # purple
        '#4DBD33',  # green
        '#87421F',  # brown
        '#FFFFFA',  # black
        '#000000',  # white
        '#00FF00',  #
    ])

    label_color = [COLORS[index] for index, _ in enumerate(classes)]
    plt.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", cmap=plt.cm.get_cmap("jet", 256), linewidths=1)

    plt.show()


main()