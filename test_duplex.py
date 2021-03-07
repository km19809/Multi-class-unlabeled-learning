import numpy as np
import matplotlib.pyplot as plt
import custom_layers, get_data
import keras
from keras import Model, Input
from keras.layers import Dense
from keras import layers
from keras.optimizers import SGD, Adam
from sklearn.manifold import TSNE
import tensorflow as tf
import datetime
import gc
import argparse

from keras.utils import plot_model
from IPython.display import Image

on_server = False
try:
    import winsound

    def play_sound():
        frequency = 1760  # Set Frequency To 2500 Hertz
        duration = 900  # Set Duration To 1000 ms == 1 second
        winsound.Beep(frequency, duration)
except ImportError:
    on_server = True

    def play_sound():
        pass


# tf.compat.v1.disable_eager_execution()
print("Version rf:", tf.__version__)
print("Eager:", tf.executing_eagerly())

np.random.seed(0)


# classi del problema
positive_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
negative_classes = [9]

classes = None
num_classes = None
num_pos_classes = None


def get_dataset():
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = get_data.get_data(positive_classes,negative_classes,
                                                                               perc_labeled, flatten_data=not use_convolutional, perc_size=perc_ds,
                                                                                       dataset_name=dataset_name)

    # esigenze per la loss
    if len(ds_labeled) % batch_size_labeled != 0:
        ds_labeled = ds_labeled[:-(len(ds_labeled) % batch_size_labeled)]
        y_labeled = y_labeled[:-(len(y_labeled) % batch_size_labeled)]

    # esigenze per la loss
    if len(ds_unlabeled) % batch_size_labeled != 0:
        ds_unlabeled = ds_unlabeled[:-(len(ds_unlabeled) % batch_size_labeled)]
        y_unlabeled = y_unlabeled[:-(len(y_unlabeled) % batch_size_labeled)]

    return ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val


def plot_2d(x, y, y_true, centroids, show_fig=False, perc_to_compute=0.15):

    cmap = plt.cm.get_cmap("jet", 256)

    x_for_tsne = np.array([t for i, t in enumerate(x) if i < len(x) * perc_to_compute])
    y_for_tsne = np.array([t for i, t in enumerate(y) if i < len(x) * perc_to_compute])
    y_true_for_tsne = np.array([t for i, t in enumerate(y_true) if i < len(x) * perc_to_compute])

    # get data in 2D
    x_embedded = TSNE(n_components=2, verbose=0).fit_transform(np.concatenate((x_for_tsne, centroids), axis=0))
    vis_x = x_embedded[:-len(centroids), 0]
    vis_y = x_embedded[:-len(centroids), 1]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Predicted vs True')

    # predicted
    ax1.scatter(vis_x, vis_y, c=y_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.2)

    # true
    ax2.scatter(vis_x, vis_y, c=y_true_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.2)

    # centroids
    label_color = [index for index, _ in enumerate(centroids)]
    ax1.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

    ax2.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

    # color bar
    norm = plt.cm.colors.Normalize(vmax=num_classes - 1, vmin=0)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)

    path = 'images/clusters_tsne_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.png'
    plt.savefig(path)

    if not on_server:
        print("Plotting...", path)

        play_sound()
        if show_fig:
            plt.show()


def create_autoencoder(input_shape, act='relu', init='glorot_uniform'):

    if use_convolutional:
        filters = [32, 64, 128, 10]

        if input_shape[0] % 8 == 0:
            pad3 = 'same'
        else:
            pad3 = 'valid'

        input_data = Input(shape=input_shape, name='input')

        if use_3d:
            # convoluzione RGB
            e = layers.Reshape((input_shape[0], input_shape[1], input_shape[2], 1))(input_data)
            e = layers.Conv3D(filters[0], (5, 5, input_shape[-1]), strides=(2, 2, input_shape[-1]), padding='same', activation='relu', name='conv1',
                              input_shape=input_shape)(e)
            e = layers.Reshape((int(input_shape[0] / 2), int(input_shape[1] / 2), filters[0]))(e)
        else:
            e = layers.Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_data)

        e = layers.Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(e)
        e = layers.Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(e)
        e = layers.Flatten()(e)
        e = layers.Dense(units=filters[3], name='embedding')(e)

        encoder_model = keras.Model(input_data, e)

        d = layers.Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu')(e)
        d = layers.Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]))(d)
        d = layers.Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(d)
        d = layers.Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(d)

        if use_3d:
            # deconv rgb
            d = layers.Reshape((int(input_shape[0] / 2), int(input_shape[1] / 2), 1, filters[0]))(d)
            d = layers.Conv3DTranspose(1, (5, 5, input_shape[-1]), strides=(2, 2, input_shape[-1]), padding='same', name='deconv1')(d)
            d = layers.Reshape(input_shape)(d)
        else:
            d = layers.Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(d)

        autoencoder_model = keras.Model(input_data, d)
    else:

        dims = [input_shape, 500, 500, 2000, 10]
        n_stacks = len(dims) - 1

        input_data = Input(shape=dims[0], name='input')
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
        x = Dense(dims[0][0], kernel_initializer=init, name='decoder_0')(x)

        decoded = x

        autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')
        encoder_model = Model(inputs=input_data, outputs=encoded, name='encoder')

    return autoencoder_model, encoder_model


def run_duplex(model_unlabeled, model_labeled, encoder, clustering_layer,
               ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, kld_weight=0.1,
               mse_weight=1, maxiter=10000):

    batch_size_unlabeled = 256
    miniter = 200
    tol = 0.001  # tolerance threshold to stop training

    print("Beginning training, maxiter:", maxiter, ", tol:", tol)

    y_pred_last = None
    all_x = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
    all_y = np.concatenate((y_unlabeled, y_labeled), axis=0)

    # ci si assicura un equo processamento di esempi etichettati e non
    labeled_interval = max(1, int(((1 / perc_labeled) - 1) * (batch_size_labeled / batch_size_unlabeled)))
    print("labeled_interval", labeled_interval)
    print("batch_size_labeled", batch_size_labeled)

    # compile models
    model_unlabeled.compile(loss=['kld', 'mse'],
                            loss_weights=[kld_weight, mse_weight], optimizer=Adam())
    model_labeled.compile(loss=[custom_layers.get_my_argmax_loss(batch_size_labeled, ce_function_type, m_prod_type=m_prod_type)],
                          loss_weights=[gamma_ce], optimizer=Adam())

    # bisogna avere anche le etichette per i negativi (tutte impostate a zero)
    temp_y_for_model_labeled = keras.utils.to_categorical(y_labeled)
    temp_y_for_model_labeled = temp_y_for_model_labeled[:, positive_classes]

    num_clusters = len(clustering_layer.get_centroids())
    y_for_model_labeled = np.zeros((temp_y_for_model_labeled.shape[0], num_clusters))

    remaining_elements = num_clusters - len(positive_classes)
    if remaining_elements > 0:
        y_for_model_labeled[:, :-remaining_elements] = temp_y_for_model_labeled
    del temp_y_for_model_labeled
    # fine codice boiler

    loss = -1
    index_unlabeled = 0
    index_labeled = 0

    for ite in range(int(maxiter)):

        # show data each epoch
        if ite % (int(len(all_x) / batch_size_unlabeled) * 5) == 0:
            y_pred_p, _ = model_unlabeled.predict(all_x, verbose=0)
            y_pred_p = y_pred_p.argmax(1)
            centroids = clustering_layer.get_centroids()

            plot_2d(encoder.predict(all_x), y_pred_p, all_y, centroids)
            del y_pred_p

        # labeled training (questo metodo funziona fintantoche gli esempi labeled sono meno degli unlabeled)
        if ite % labeled_interval == 0:
            if (index_labeled + 1) * batch_size_labeled > ds_labeled.shape[0]:
                index_labeled = 0

            loss = model_labeled.train_on_batch(
                x=ds_labeled[index_labeled * batch_size_labeled:(index_labeled + 1) * batch_size_labeled],
                y=[y_for_model_labeled[index_labeled * batch_size_labeled:(index_labeled + 1) * batch_size_labeled]
                   ])
            index_labeled += 1

        if ite % int(len(ds_labeled) / batch_size_labeled) == 0:
            # shuffle data (experimental)
            shuffler1 = np.random.permutation(len(ds_labeled))
            ds_labeled = ds_labeled[shuffler1]
            y_labeled = y_labeled[shuffler1]
            y_for_model_labeled = y_for_model_labeled[shuffler1]

            del shuffler1

        # update target probability
        if ite % update_interval == 0:
            q, _ = model_unlabeled.predict(all_x, verbose=0)
            p = custom_layers.target_distribution(q)  # update the auxiliary target distribution p
            y_pred_u = q.argmax(1)

            if all_y is not None:
                # evaluate the clustering performance
                custom_layers.print_measures(all_y, y_pred_u, classes, ite=ite)
                #print('Loss=', np.round(loss, 5))

            # check stop criterion
            delta_label = np.sum(y_pred_u != y_pred_last).astype(np.float32) / y_pred_u.shape[0] if y_pred_last is not None else 1
            y_pred_last = y_pred_u
            if ite > miniter and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                'Reached tolerance threshold. Stopping training.'
                break

            del y_pred_u
            del q

        # unlabeled train on batch
        if (index_unlabeled + 1) * batch_size_unlabeled > all_x.shape[0]:
            loss = model_unlabeled.train_on_batch(x=all_x[index_unlabeled * batch_size_unlabeled::],
                                                  y=[p[index_unlabeled * batch_size_unlabeled::],
                                                     all_x[index_unlabeled * batch_size_unlabeled::]])
            index_unlabeled = 0
        else:
            loss = model_unlabeled.train_on_batch(x=all_x[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled],
                                             y=[p[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled],
                                                all_x[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled]])
            index_unlabeled += 1

        #print('Iter', ite, "UNlabeled loss is", loss)

        if ite % 1000 == 0:
            gc.collect()


def main():

    # print dei parametri
    print(" ------------------------------------------- ")
    print("ce_function_type", ce_function_type, "m_prod_type", m_prod_type, "gamma_ce", gamma_ce, "gamma_kld", gamma_kld,
          "update_interval", update_interval, "batch_size_labeled", batch_size_labeled, "init_kmeans", init_kmeans)
    print("use_convolutional", use_convolutional, "perc_ds", perc_ds, "perc_labeled", perc_labeled, "dataset_name", dataset_name)
    print("positive_classes", positive_classes, "negative_classes", negative_classes)

    # dataset
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = get_dataset()

    all_ds = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
    all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)

    # PRETRAINING autoencoder
    batch_size = 256

    autoencoder, encoder = create_autoencoder(all_ds[0].shape)
    autoencoder.compile(optimizer=Adam(), loss='mse')

    if not on_server:
        plot_model(autoencoder, to_file='images/_model_autoencoder.png', show_shapes=True)
        Image(filename='images/_model_autoencoder.png')

    # TRAINING (se i parametri sono stati già salvati, li si prende da file system)
    model_loaded = False

    name_file_model = 'parameters/' + dataset_name + '_duplex_pretraining_' + ('conv' if use_convolutional else 'fwd')

    try:
        autoencoder.load_weights(name_file_model)
        model_loaded = True
    except Exception:
        pass

    if not model_loaded:
        print("Training autoencoder...")
        autoencoder.fit(all_ds, all_ds, batch_size=batch_size, epochs=autoencoder_n_epochs, shuffle=True)
        autoencoder.save_weights(name_file_model)

    # show dataset
    if False:
        for i in range(9):
            plt.subplot(330+1+i)

            if i % 2 == 0:
                plt.imshow(all_ds[i])
            else:
                rec = autoencoder.predict(all_ds[i-1:i])[0]
                plt.imshow(rec)
        plt.show()

    # FINE AUTOENCODER

    # INIZIO PRETRAINING SUPERVISIONATO
    # prima si allena con solo i centroidi positivi

    # run k means for cluster centers
    # centroids = custom_layers.get_centroids_from_kmeans(num_pos_classes, positive_classes, ds_unlabeled, ds_labeled, y_labeled, encoder, init_kmeans=init_kmeans)
    centroids = custom_layers.compute_centroids_from_labeled(encoder, ds_labeled, y_labeled, positive_classes)

    # last layer
    clustering_layer = custom_layers.ClusteringLayer(num_pos_classes, weights=[centroids], name='clustering')
    unlabeled_last_layer = clustering_layer(encoder.output)
    labeled_last_layer = keras.layers.Softmax()(unlabeled_last_layer)

    # models
    model_unlabeled = Model(inputs=encoder.input, outputs=[unlabeled_last_layer, autoencoder.output])
    model_labeled = Model(inputs=encoder.input, outputs=[labeled_last_layer])

    if not on_server:
        plot_model(model_unlabeled, to_file='images/model_unlabeled.png', show_shapes=True)
        Image(filename='images/model_unlabeled.png')
        plot_model(model_labeled, to_file='images/model_labeled.png', show_shapes=True)
        Image(filename='images/model_labeled.png')

    # train
    #model_unlabeled.load_weights("parameters/" + dataset_name + "_duplex_pretraining2_unlabeled")
    #model_labeled.load_weights("parameters/" + dataset_name + "_duplex_pretraining2_labeled")

    run_duplex(model_unlabeled, model_labeled, encoder, clustering_layer, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, kld_weight=0)
    model_unlabeled.save_weights("parameters/" + dataset_name + "_duplex_pretraining2_unlabeled")
    model_labeled.save_weights("parameters/" + dataset_name + "_duplex_pretraining2_labeled")

    #centroids = [c for c in clustering_layer.get_centroids()] #utilizzati per il prossimo k means
    centroids = []

    # FINE ALLENAMENTO SUP

    # CUSTOM TRAINING (tutte le classi)
    # run k means for cluster centers
    centroids = custom_layers.get_centroids_from_kmeans(num_classes, positive_classes, ds_unlabeled, ds_labeled,
                                                           y_labeled, encoder, init_kmeans=init_kmeans, centroids=centroids)

    clustering_layer = custom_layers.ClusteringLayer(num_classes, weights=[centroids], name='clustering')

    # last layers
    unlabeled_last_layer = clustering_layer(encoder.output)
    labeled_last_layer = keras.layers.Softmax()(unlabeled_last_layer)

    # models
    model_unlabeled = Model(inputs=encoder.input, outputs=[unlabeled_last_layer, autoencoder.output])
    model_labeled = Model(inputs=encoder.input, outputs=[labeled_last_layer])

    if not on_server:
        plot_model(model_unlabeled, to_file='images/model_unlabeled.png', show_shapes=True)
        Image(filename='images/model_unlabeled.png')
        plot_model(model_labeled, to_file='images/model_labeled.png', show_shapes=True)
        Image(filename='images/model_labeled.png')

    # fit
    if True:

        run_duplex(model_unlabeled, model_labeled, encoder, clustering_layer, ds_labeled, y_labeled,
                   ds_unlabeled, y_unlabeled, kld_weight=gamma_kld)

        model_unlabeled.save_weights("parameters/" + dataset_name + "_duplex_trained_unlabeled")
        model_labeled.save_weights("parameters/" + dataset_name + "_duplex_trained_labeled")
    else:
        model_unlabeled.load_weights("parameters/" + dataset_name + "_duplex_trained_unlabeled")
        model_labeled.load_weights("parameters/" + dataset_name + "_duplex_trained_labeled")

    # FINE TRAINING

    # accuratezza sul dataset di training
    y_pred = model_unlabeled.predict(all_ds, verbose=0)[0].argmax(1)
    x_embedded_encoder = encoder.predict(all_ds)
    custom_layers.print_measures(all_y, y_pred, classes, x_for_silouhette=x_embedded_encoder)

    # plot
    centroids = clustering_layer.get_centroids()
    plot_2d(x_embedded_encoder, y_pred, all_y, centroids, perc_to_compute=0.8)

    # VALIDATION DATA
    print("Test on VALIDATION DATA")

    # accuratezza
    y_pred = model_unlabeled.predict(x_val, verbose=0)[0].argmax(1)
    x_embedded_encoder = encoder.predict(x_val)
    custom_layers.print_measures(y_val, y_pred, classes, x_for_silouhette=x_embedded_encoder)

    # plot
    plot_2d(x_embedded_encoder, y_pred, y_val, centroids, perc_to_compute=1)


# parametri per il training
use_convolutional = True
use_3d = True
perc_labeled = 0.15
perc_ds = 1
dataset_name = 'cifar'

# iperparametri del modello
autoencoder_n_epochs = 200
batch_size_labeled = 150
gamma_kld = 0.1
gamma_ce = 0.1
ce_function_type = "all"
m_prod_type = "molt"
update_interval = 140
init_kmeans = True
do_suite_test = True


def read_args():
    # Si specificano i parametri in input che modificano il comportamento del sistema
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_convolutional')
    parser.add_argument('--perc_labeled')
    parser.add_argument('--perc_ds')
    parser.add_argument('--dataset_name')
    parser.add_argument('--batch_size_labeled')

    parser.add_argument('--gamma_kld')
    parser.add_argument('--gamma_ce')
    parser.add_argument('--ce_function_type')
    parser.add_argument('--m_prod_type')
    parser.add_argument('--update_interval')

    parser.add_argument('--init_kmeans')
    parser.add_argument('--do_suite_test')
    parser.add_argument('--positive_classes')
    parser.add_argument('--negative_classes')

    args = parser.parse_args()

    global use_convolutional, perc_labeled, perc_ds, dataset_name, batch_size_labeled, gamma_kld, gamma_ce,\
        ce_function_type, m_prod_type, update_interval, init_kmeans, do_suite_test, positive_classes, negative_classes

    if args.use_convolutional:
        use_convolutional = args.use_convolutional == 'True'
    if args.perc_labeled:
        perc_labeled = float(args.perc_labeled)
    if args.perc_ds:
        perc_ds = float(args.perc_ds)
    if args.dataset_name:
        dataset_name = args.dataset_name
    if args.batch_size_labeled:
        batch_size_labeled = int(args.batch_size_labeled)

    if args.gamma_kld:
        gamma_kld = float(args.gamma_kld)
    if args.gamma_ce:
        gamma_ce = float(args.gamma_ce)
    if args.ce_function_type:
        ce_function_type = args.ce_function_type
    if args.m_prod_type:
        m_prod_type = args.m_prod_type
    if args.update_interval:
        update_interval = int(args.update_interval)

    if args.init_kmeans:
        init_kmeans = args.init_kmeans == 'True'
    if args.do_suite_test:
        do_suite_test = args.do_suite_test == 'True'
    if args.positive_classes:
        positive_classes = []
        for s in args.positive_classes.split(','):
            positive_classes.append(int(s))
    if args.negative_classes:
        negative_classes = []
        for s in args.negative_classes.split(','):
            negative_classes.append(int(s))

    # parametri calcolati
    global classes, num_classes, num_pos_classes
    classes = positive_classes.copy()
    classes.extend(negative_classes)

    num_classes = len(classes)
    num_pos_classes = len(positive_classes)


# lettura parametri
read_args()
if do_suite_test:
    for i in classes:
        positive_classes = [c for c in classes if c != i]
        negative_classes = [c for c in classes if c == i]
        main()
else:
    main()

