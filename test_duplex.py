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

print(tf.__version__)
print(tf.executing_eagerly())

np.random.seed(0)


positive_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
negative_classes = [9]

classes = positive_classes.copy()
classes.extend(negative_classes)

num_classes = len(classes)
num_pos_classes = len(positive_classes)

use_convolutional = True
perc_labeled = 0.05
batch_size_labeled = 150

dataset_name = 'cifar'


def get_dataset():
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = get_data.get_data(positive_classes,negative_classes,
                                                                               perc_labeled, flatten_data=not use_convolutional, perc_size=1,
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


def plot_2d(x, y, y_true, centroids, show_fig=False, perc_to_compute=0.2):

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

    if not on_server:
        print("Plotting...", path)
        plt.savefig(path)

        play_sound()
        if show_fig:
            plt.show()


def create_autoencoder(dims, act='relu', init='glorot_uniform'):

    if dataset_name == "cifar":

        input_shape = (32, 32, 3)
        filters = [32, 64, 128, 10]

        model = keras.Sequential()
        if input_shape[0] % 8 == 0:
            pad3 = 'same'
        else:
            pad3 = 'valid'

        input_data = Input(shape=dims[0], name='input')
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
        d = layers.Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(d)

        autoencoder_model = keras.Model(input_data, d)
    else:
        if use_convolutional:
            input_data = Input(shape=dims[0], name='input')

            # ENCODER
            e = layers.Conv2D(32, (3, 3), activation='relu')(input_data)
            e = layers.MaxPooling2D((2, 2))(e)
            e = layers.Conv2D(64, (3, 3), activation='relu')(e)
            e = layers.MaxPooling2D((2, 2))(e)
            e = layers.Conv2D(64, (3, 3), activation='relu')(e)
            l = layers.Flatten()(e)
            l = layers.Dense(49, activation=act)(l)
            l = layers.Dense(10, kernel_initializer=init)(l)

            encoder_model = keras.Model(input_data, l)

            # DECODER
            d = layers.Dense(49, activation=act)(l)
            d = layers.Reshape((7, 7, 1))(d)
            d = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
            d = layers.BatchNormalization()(d)
            d = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
            d = layers.BatchNormalization()(d)
            d = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(d)
            d = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

            autoencoder_model = keras.Model(input_data, d)
        else:
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
    y_pred_last = None
    all_x = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
    all_y = np.concatenate((y_unlabeled, y_labeled), axis=0)

    batch_size_unlabeled = 256
    miniter = 200
    tol = 0.005  # tolerance threshold to stop training

    # ci si assicura un equo processamento di esempi etichettati e non
    labeled_interval = int(((1 / perc_labeled) - 1) * (batch_size_labeled / batch_size_unlabeled))
    print("labeled_interval", labeled_interval)
    print("batch_size_labeled", batch_size_labeled)

    #update_interval = 140
    #update_interval = labeled_interval

    # compile models
    model_unlabeled.compile(loss=['kld', 'mse'],
                            loss_weights=[kld_weight, mse_weight], optimizer=Adam())
    model_labeled.compile(loss=[custom_layers.get_my_argmax_loss(batch_size_labeled, ce_function_type, m_prod_type=m_prod_type)],
                          loss_weights=[gamma_ce], optimizer=Adam())

    # bisogna avere anche le etichette per i negativi
    y_labeled = np.concatenate((y_labeled, [num_classes - 1]), axis=0)
    temp_y_for_model_labeled = keras.utils.to_categorical(y_labeled)
    y_labeled = y_labeled[:-1]
    temp_y_for_model_labeled = temp_y_for_model_labeled[:-1]

    temp_y_for_model_labeled = np.delete(temp_y_for_model_labeled, negative_classes, axis=1)

    num_clusters = len(clustering_layer.get_centroids())
    y_for_model_labeled = np.empty((temp_y_for_model_labeled.shape[0], num_clusters))
    rm_zeros = np.zeros(num_clusters - temp_y_for_model_labeled.shape[1])

    for i, el in enumerate(temp_y_for_model_labeled):
        y_for_model_labeled[i] = np.concatenate((el, rm_zeros), axis=0)
    del temp_y_for_model_labeled
    #fine codice boiler

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
    print("ce_function_type", ce_function_type, "gamma_ce", gamma_ce, "gamma_kld", gamma_kld, "update_interval",
          update_interval, "init_kmeans", init_kmeans)
    print("positive_classes", positive_classes)
    print("negative_classes", negative_classes)

    # dataset
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = get_dataset()

    all_ds = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
    all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)

    # PRETRAINING autoencoder
    batch_size = 256
    dims = [all_ds[0].shape, 500, 500, 2000, 10]

    autoencoder, encoder = create_autoencoder(dims)
    autoencoder.compile(optimizer=Adam(), loss='mse')

    if not on_server:
        plot_model(autoencoder, to_file='images/_model_autoencoder.png', show_shapes=True)
        Image(filename='images/_model_autoencoder.png')

    # TRAINING (se i parametri sono stati già salvati, li si prende da file system)
    model_loaded = False

    name_file_model = 'parameters/duplex_pretraining_conv' if use_convolutional else 'parameters/duplex_pretraining'

    try:
        autoencoder.load_weights(name_file_model)
        model_loaded = True
    except Exception:
        pass

    if not model_loaded:
        print("Training autoencoder...")
        n_epochs = 200
        autoencoder.fit(all_ds, all_ds, batch_size=batch_size, epochs=n_epochs, shuffle=True)
        autoencoder.save_weights(name_file_model)

    # show dataset
    for i in range(9):
        plt.subplot(330+1+i)

        if i % 2 == 0:
            plt.imshow(all_ds[i])
        else:
            rec = autoencoder.predict(all_ds[i:i+1])[0]
            plt.imshow(rec)

    plt.show()

    # experimental
    # prima si allena con solo i centroidi positivi
    # CUSTOM TRAINING
    clustering_layer = custom_layers.ClusteringLayer(num_pos_classes, name='clustering')

    # last layer
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

    # run k means for cluster centers
    #_, centroids = custom_layers.get_centroids_from_kmeans(num_pos_classes, positive_classes, ds_unlabeled, ds_labeled, y_labeled, encoder, init_kmeans=init_kmeans)
    centroids = custom_layers.compute_centroids_from_labeled(encoder, ds_labeled, y_labeled, positive_classes)

    model_unlabeled.get_layer(name='clustering').set_weights([centroids])

    #model_unlabeled.load_weights("parameters/" + dataset_name + "_duplex_pretraining2_unlabeled")
    #model_labeled.load_weights("parameters/" + dataset_name + "_duplex_pretraining2_labeled")

    run_duplex(model_unlabeled, model_labeled, encoder, clustering_layer, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, kld_weight=0)
    model_unlabeled.save_weights("parameters/" + dataset_name + "_duplex_pretraining2_unlabeled")
    model_labeled.save_weights("parameters/" + dataset_name + "_duplex_pretraining2_labeled")

    #centroids = [c for c in clustering_layer.get_centroids()] #utilizzati per il prossimo k means
    centroids = []
    # fine prima parte


    # CUSTOM TRAINING (tutte le classi)
    clustering_layer = custom_layers.ClusteringLayer(num_classes, name='clustering')

    # last layer
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

    # run k means for cluster centers
    _, centroids = custom_layers.get_centroids_from_kmeans(num_classes, positive_classes, ds_unlabeled, ds_labeled, y_labeled,
                                                                encoder, init_kmeans=init_kmeans, centroids=centroids)

    model_unlabeled.get_layer(name='clustering').set_weights([centroids])

    # fit
    if True:

        run_duplex(model_unlabeled, model_labeled, encoder, clustering_layer, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, kld_weight=gamma_kld)


        model_unlabeled.save_weights("parameters/" + dataset_name + "_duplex_trained_unlabeled")
        model_labeled.save_weights("parameters/" + dataset_name + "_duplex_trained_labeled")
    else:
        model_unlabeled.load_weights("parameters/" + dataset_name + "_duplex_trained_unlabeled")
        model_labeled.load_weights("parameters/" + dataset_name + "_duplex_trained_labeled")

    # accuratezza
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


# iperparametri
gamma_kld = 0.1
gamma_ce = 0.1
ce_function_type = "all"
m_prod_type = "molt"
update_interval = 140
init_kmeans = True
do_suite_test = True


if not do_suite_test:
    main()
else:
    for i in range(num_classes):
        positive_classes = [c for c in classes if c != i]
        negative_classes = [c for c in classes if c == i]
        main()
