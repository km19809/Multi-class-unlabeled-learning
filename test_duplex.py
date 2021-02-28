import numpy as np
import matplotlib.pyplot as plt
import custom_layers, get_data
import keras
from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
import tensorflow as tf

from keras.utils import plot_model
from IPython.display import Image

print(tf.__version__)
print(tf.executing_eagerly())

np.random.seed(0)

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
positive_classes = [0, 1, 2, 3, 4, 5, 6, ]
negative_classes = [7, 8, 9]

num_classes = len(classes)
num_pos_classes = len(positive_classes)

#num_dim = 2

#total_samples_per_class = 256
perc_labeled = 0.1

batch_size_labeled = 200


def get_dataset():
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, _, _ = get_data.get_data(positive_classes,negative_classes,perc_labeled,flatten_data=True)

    # esigenze per la loss
    if len(ds_labeled) % batch_size_labeled != 0:
        ds_labeled = ds_labeled[:-(len(ds_labeled) % batch_size_labeled)]
        y_labeled = y_labeled[:-(len(y_labeled) % batch_size_labeled)]

    # esigenze per la loss
    if len(ds_unlabeled) % batch_size_labeled != 0:
        ds_unlabeled = ds_unlabeled[:-(len(ds_unlabeled) % batch_size_labeled)]
        y_unlabeled = y_unlabeled[:-(len(y_unlabeled) % batch_size_labeled)]

    return ds_labeled, y_labeled, ds_unlabeled, y_unlabeled


def plot_2d(x, y, centroids):

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

    from sklearn.manifold import TSNE
    x_embedded = TSNE(n_components=2, verbose=1).fit_transform(np.concatenate((x, centroids), axis=0))
    vis_x = x_embedded[:-len(centroids), 0]
    vis_y = x_embedded[:-len(centroids), 1]

    plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 256), linewidths=0.2, marker=".", alpha=0.1)
    plt.colorbar(ticks=range(256))
    #plt.xlim((-11, 11))
    #plt.ylim((-11,11))

    # scatter centroids
    label_color = [COLORS[index] for index in range(num_classes)]
    plt.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", cmap=plt.cm.get_cmap("jet", 256), linewidths=1)

    plt.show()


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


def print_accuracy(x, y, centroids, label, model, encoder):

    mapping_indexes = dict()

    for y_class in range(num_classes):

        only_x_class, _ = get_data.filter_ds(x, y, [y_class])
        only_x_class = encoder.predict(only_x_class)

        centroid_class = np.mean(only_x_class, axis=0)

        index_nearest_centroid = np.argmin([np.linalg.norm(centroid - centroid_class) for centroid in centroids])

        mapping_indexes[index_nearest_centroid] = y_class

    q = model.predict(x, verbose=0)

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


def run_inter(model, x, y, y_pred_last):

    batch_size = 256
    maxiter = 10000
    miniter = 5000
    update_interval = 140
    tol = 0.0001  # tolerance threshold to stop training

    model.compile(loss='kld', optimizer=Adam())

    loss = [0, 0, 0]
    index = 0
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(x, verbose=0)
            p = custom_layers.target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if y is not None:
                acc = np.round(cluster_acc(y, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                loss = np.round(loss, 5)

                print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            if ite > miniter and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                'Reached tolerance threshold. Stopping training.'
                break

        # train on batch
        if (index + 1) * batch_size > x.shape[0]:
            loss = model.train_on_batch(x=x[index * batch_size::],y=p[index * batch_size::])
            index = 0
        else:
            loss = model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                             y=p[index * batch_size:(index + 1) * batch_size])
            index += 1


def run_argmax(model, x, y):

    model.compile( optimizer=Adam(), loss=custom_layers.get_my_argmax_loss(batch_size_labeled))

    y_for_model = keras.utils.to_categorical(y)

    model.fit(x, y_for_model,
                      epochs=40,
                      batch_size=batch_size_labeled,
                      shuffle=True)


def run_duplex(model_unlabeled, model_labeled, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, y_pred_last):

    batch_size_unlabeled = 256
    maxiter = 10000
    miniter = 200
    tol = 0.01  # tolerance threshold to stop training

    epochs_labeled = 5
    labeled_interval = int((len(ds_unlabeled) / batch_size_unlabeled) * epochs_labeled)  #every N epochs do labeled training
    print("labeled_interval", labeled_interval)

    #update_interval = 500
    update_interval = labeled_interval

    # compile models
    model_unlabeled.compile(loss='kld', optimizer=Adam())
    model_labeled.compile(optimizer=Adam(), loss=custom_layers.get_my_argmax_loss(batch_size_labeled))

    # bisogna avere anche le etichette per i negativi
    temp_y_for_model_labeled = keras.utils.to_categorical(y_labeled)
    y_for_model_labeled = np.empty((temp_y_for_model_labeled.shape[0], num_classes))

    rm_zeros = np.zeros(num_classes - temp_y_for_model_labeled.shape[1])
    for i, el in enumerate(temp_y_for_model_labeled):
        y_for_model_labeled[i] = np.concatenate((el, rm_zeros), axis=0)

    loss = -1
    index = 0
    for ite in range(int(maxiter)):

        # labeled training
        if ite % labeled_interval == 0:
            print("Labeled training:", ite)
            model_labeled.fit(ds_labeled, y_for_model_labeled, verbose=0,
                      epochs=epochs_labeled, batch_size=batch_size_labeled, shuffle=True)

        # update target probability
        if ite % update_interval == 0:
            q = model_unlabeled.predict(ds_unlabeled, verbose=0)
            p = custom_layers.target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if y_unlabeled is not None:
                acc = np.round(cluster_acc(y_unlabeled, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y_unlabeled, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y_unlabeled, y_pred), 5)
                loss = np.round(loss, 5)

                print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            if ite > miniter and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                'Reached tolerance threshold. Stopping training.'
                break

        # unlabeled train on batch
        if (index + 1) * batch_size_unlabeled > ds_unlabeled.shape[0]:
            loss = model_unlabeled.train_on_batch(x=ds_unlabeled[index * batch_size_unlabeled::],y=p[index * batch_size_unlabeled::])
            index = 0
        else:
            loss = model_unlabeled.train_on_batch(x=ds_unlabeled[index * batch_size_unlabeled:(index + 1) * batch_size_unlabeled],
                                             y=p[index * batch_size_unlabeled:(index + 1) * batch_size_unlabeled])
            index += 1




def main():
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled = get_dataset()

    all_ds = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
    all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)

    # PRETRAINING autoencoder
    n_epochs = 50
    batch_size = 256
    dims = [all_ds.shape[-1], 500, 500, 2000, 10]

    autoencoder, encoder = create_autoencoder(dims)
    autoencoder.compile(optimizer=Adam(), loss='mse')

    # TRAINING (se i parametri sono stati già salvati, li si prende da file system)
    model_loaded = False
    name_file_model = 'parameters/duplex_pretraining'

    try:
        autoencoder.load_weights(name_file_model)
        model_loaded = True
    except Exception:
        pass

    if not model_loaded:
        autoencoder.fit(all_ds, all_ds, batch_size=batch_size, epochs=n_epochs)
        autoencoder.save_weights(name_file_model)


    # CUSTOM TRAINING
    clustering_layer = custom_layers.ClusteringLayer(num_classes, name='clustering')

    # last layer
    unlabeled_last_layer = clustering_layer(encoder.output)
    labeled_last_layer = keras.layers.Softmax()(unlabeled_last_layer)

    # models
    model_unlabeled = Model(inputs=encoder.input, outputs=unlabeled_last_layer)
    model_labeled = Model(inputs=encoder.input, outputs=labeled_last_layer)

    plot_model(model_unlabeled, to_file='model_unlabeled.png', show_shapes=True)
    Image(filename='model_unlabeled.png')
    plot_model(model_labeled, to_file='model_labeled.png', show_shapes=True)
    Image(filename='model_labeled.png')

    # run k means for cluster centers
    centroids = []
    for y_class in positive_classes:
        only_x_class, _ = get_data.filter_ds(ds_labeled, y_labeled, [y_class])
        centroids.append(np.mean(encoder.predict(only_x_class), axis=0))
    while len(centroids) < num_classes:
        centroids.append(np.random.normal(np.mean(centroids, axis=0), np.std(centroids, axis=0)))
    centroids = np.array(centroids)

    kmeans = KMeans(n_clusters=num_classes, init=centroids)
    y_pred = kmeans.fit_predict(encoder.predict(all_ds))
    model_unlabeled.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])


    # fit
    y_pred_last = np.copy(y_pred)

    #run_inter(model_unlabeled, all_ds, all_y, y_pred_last)
    #run_argmax(model_labeled, all_ds, all_y)
    run_duplex(model_unlabeled, model_labeled, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, y_pred_last)

    model_unlabeled.save_weights("parameters/11")
    model_labeled.save_weights("parameters/22")

    # accuratezza
    centroids = clustering_layer.get_centroids()
    y_pred = print_accuracy(all_ds, all_y, centroids, "after DEC", model_unlabeled, encoder)

    # silhouette
    x_embedded_encoder = encoder.predict(all_ds)
    score = silhouette_score(x_embedded_encoder, y_pred, metric='euclidean')
    print("Silouhette score:" + str(score))

    # plot
    plot_2d(x_embedded_encoder, y_pred, centroids)







main()