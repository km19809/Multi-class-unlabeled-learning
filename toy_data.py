import numpy as np
import matplotlib.pyplot as plt
import custom_layers, get_data
from keras import Model, Input
from keras.optimizers import SGD, Adam
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment

np.random.seed(0)


num_classes = 4
num_pos_classes = 3

num_dim = 2

total_samples_per_class = 100
perc_labeled = 0.5


def get_dataset():
    # restituisce dataset etichettati e non etichettati

    ds_labeled = np.empty((0, 2))
    y_labeled = np.empty(0)
    ds_unlabeled = np.empty((0, 2))
    y_unlabeled = np.empty(0)

    for index_class in range(num_classes):
        centroid = np.random.normal([0.0, 0.0], 5)
        samples = np.random.normal(centroid, 0.5, (total_samples_per_class, 2))

        if index_class < num_pos_classes:
            n_labeled = total_samples_per_class * perc_labeled
        else:
            n_labeled = 0 # classe totalmente non etichettata

        samples_labeled = np.array([s for i, s in enumerate(samples) if i < n_labeled])
        samples_unlabeled = np.array([s for i, s in enumerate(samples) if i >= n_labeled])

        if n_labeled > 0:
            ds_labeled = np.concatenate((ds_labeled, samples_labeled), axis=0)
            y_labeled = np.concatenate((y_labeled, [index_class for _ in samples_labeled]), axis=0)

        ds_unlabeled = np.concatenate((ds_unlabeled, samples_unlabeled), axis=0)
        y_unlabeled = np.concatenate((y_unlabeled, [index_class for _ in samples_unlabeled]), axis=0)

    return ds_labeled, y_labeled, ds_unlabeled, y_unlabeled


def plot_2d(x, y, centroids):
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 256), linewidths=0.2, marker=".")
    plt.colorbar(ticks=range(256))
    plt.xlim((-11, 11))
    plt.ylim((-11,11))

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

    label_color = [COLORS[index] for index in range(num_classes)]
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", cmap=plt.cm.get_cmap("jet", 256), linewidths=1)

    plt.show()


def create_autoencoder(dims = 2):

    input_data = Input(shape=(dims,), name='input')
    x = input_data

    encoder_model = Model(inputs=input_data, outputs=input_data, name='encoder')

    return encoder_model


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
    update_interval = 140
    tol = 0.001  # tolerance threshold to stop training

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
            if ite > 0 and delta_label < tol:
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


def main():
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled = get_dataset()

    all_ds = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
    all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)

    #plot_2d(ds_unlabeled, y_unlabeled)
    #plot_2d(ds_labeled, y_labeled)

    # DEC
    encoder = create_autoencoder()

    clustering_layer = custom_layers.ClusteringLayer(num_classes, name='clustering')
    cl = clustering_layer(encoder.output)
    model = Model(inputs=encoder.input, outputs=cl)

    # run k means for cluster centers
    kmeans = KMeans(n_clusters=num_classes, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(all_ds))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    y_pred_last = np.copy(y_pred)

    # fit
    run_inter(model, all_ds, all_y, y_pred_last)

    # accuratezza
    centroids = clustering_layer.get_centroids()
    y_pred = print_accuracy(all_ds, all_y, centroids, "after DEC", model, encoder)

    # silhouette
    x_embedded_encoder = encoder.predict(all_ds)
    score = silhouette_score(x_embedded_encoder, y_pred, metric='euclidean')
    print("Silouhette score:" + str(score))

    # plot
    plot_2d(x_embedded_encoder, y_pred, centroids)







main()