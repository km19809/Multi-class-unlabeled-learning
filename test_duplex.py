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
positive_classes = [1,2,3,4,5,6,7,8,9]
negative_classes = [0]

classes = None
num_classes = None
num_pos_classes = None


def get_dataset():
    flatten_data = not use_convolutional or dataset_name == "reuters"

    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = \
        get_data.get_data(positive_classes,negative_classes,
            perc_labeled, flatten_data=flatten_data, perc_size=perc_ds,
            dataset_name=dataset_name, data_preparation=dataset_name == "reuters")

    global batch_size_labeled

    # esigenze per la loss
    if len(ds_labeled) % batch_size_labeled != 0:
        ds_labeled = ds_labeled[:-(len(ds_labeled) % batch_size_labeled)]
        y_labeled = y_labeled[:-(len(y_labeled) % batch_size_labeled)]

    # esigenze per la loss
    if len(ds_unlabeled) % batch_size_labeled != 0:

        if batch_size_labeled > len(ds_unlabeled):
            # caso limite
            batch_size_labeled = len(ds_unlabeled)
        else:
            ds_unlabeled = ds_unlabeled[:-(len(ds_unlabeled) % batch_size_labeled)]
            y_unlabeled = y_unlabeled[:-(len(y_unlabeled) % batch_size_labeled)]

    return ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val


def plot_2d(x, y, y_true, index_labeled, centroids, perc_to_compute=1, show_fig=False):

    cmap = plt.cm.get_cmap("jet", 256)

    # si prende una parte dei dati (usato per velocizzare)
    shuffler1 = np.random.permutation(len(x))
    indexes_to_take = np.array([t for i, t in enumerate(shuffler1) if i < len(shuffler1) * perc_to_compute])

    x_for_tsne = x[indexes_to_take]
    y_for_tsne = y[indexes_to_take]
    y_true_for_tsne = y_true[indexes_to_take]
    labeled_for_tsne = index_labeled[indexes_to_take]

    # get data in 2D
    x_embedded = TSNE(n_components=2, verbose=0).fit_transform(np.concatenate((x_for_tsne, centroids), axis=0))
    vis_x = x_embedded[:-len(centroids), 0]
    vis_y = x_embedded[:-len(centroids), 1]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Predicted vs True , All vs Labeled')

    # ALL
    # predicted
    ax1.scatter(vis_x, vis_y, c=y_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.2)

    # true
    ax2.scatter(vis_x, vis_y, c=y_true_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.2)

    # LABELED
    labeled_samples_x = np.array([x for i, x in enumerate(vis_x) if labeled_for_tsne[i]])
    labeled_samples_y = np.array([x for i, x in enumerate(vis_y) if labeled_for_tsne[i]])
    labeled_y_for_tsne = np.array([x for i, x in enumerate(y_for_tsne) if labeled_for_tsne[i]])
    labeled_y_true_for_tsne = np.array([x for i, x in enumerate(y_true_for_tsne) if labeled_for_tsne[i]])

    # predicted
    ax3.scatter(labeled_samples_x, labeled_samples_y, c=labeled_y_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.5)

    # true
    ax4.scatter(labeled_samples_x, labeled_samples_y, c=labeled_y_true_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.5)

    # CENTROIDS
    label_color = [index for index, _ in enumerate(centroids)]
    ax1.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

    ax2.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

    ax3.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

    ax4.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1, c=label_color,
                edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

    # color bar
    norm = plt.cm.colors.Normalize(vmax=num_classes - 1, vmin=0)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3)

    path = 'images/clusters_tsne_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.png'
    plt.savefig(path)

    if not on_server:
        print("Plotting...", path)

        play_sound()
        if show_fig:
            plt.show()

    plt.close(fig)


def create_autoencoder(input_shape, act='relu', init='glorot_uniform'):

    if use_convolutional and dataset_name != "reuters":
        filters = [32, 64, 128, 10]

        if input_shape[0] % 8 == 0:
            pad3 = 'same'
        else:
            pad3 = 'valid'

        input_data = Input(shape=input_shape, name='input')
        e = layers.Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_data)
        e = layers.Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(e)
        e = layers.Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(e)
        e = layers.Flatten()(e)
        e = layers.Dense(units=filters[3], activation='tanh', name='embedding')(e)

        encoder_model = keras.Model(input_data, e)

        d = layers.Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu')(e)
        d = layers.Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]))(d)
        d = layers.Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(d)
        d = layers.Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(d)
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
        encoded = Dense(dims[-1], kernel_initializer=init, activation="tanh", #activity_regularizer=keras.regularizers.l1(0.00001),
                        name='encoder_%d' % (n_stacks - 1))(x)
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


def get_centroids(all_ds, ds_unlabeled, ds_labeled, y_labeled, encoder):
    # run k means for cluster centers
    if centroid_init == "gm":
        centroids = custom_layers.get_centroids_from_GM(num_classes, positive_classes, ds_unlabeled, ds_labeled,
                                                        y_labeled, encoder)
    elif centroid_init == "kmeans":
        centroids = custom_layers.get_centroids_from_kmeans(num_classes, positive_classes, ds_unlabeled, ds_labeled,
                                                            y_labeled, encoder)
    else:
        # si prende dalla media dei labeled
        centroids = custom_layers.compute_centroids_from_labeled(encoder, ds_labeled, y_labeled, positive_classes)
        centroids = custom_layers.get_centroids_for_clustering(encoder.predict(all_ds), num_classes, centroids)

    return centroids


def init_models(centroids, encoder, autoencoder):
    clustering_layer = custom_layers.ClusteringLayer(num_classes, weights=[centroids], name='clustering')

    # last layers
    unlabeled_last_layer = clustering_layer(encoder.output)

    if supervised_loss_type == "on_cluster":
        #labeled_last_layer = keras.layers.Softmax()(unlabeled_last_layer)
        labeled_last_layer = unlabeled_last_layer
    else:
        labeled_last_layer = encoder.output

    # models
    model_unlabeled = Model(inputs=encoder.input, outputs=[unlabeled_last_layer, autoencoder.output])
    model_labeled = Model(inputs=encoder.input, outputs=[labeled_last_layer])

    return model_unlabeled, model_labeled, clustering_layer


def plot_models(model_unlabeled, model_labeled):
    if not on_server:
        plot_model(model_unlabeled, to_file='images/model_unlabeled.png', show_shapes=True)
        Image(filename='images/model_unlabeled.png')
        plot_model(model_labeled, to_file='images/model_labeled.png', show_shapes=True)
        Image(filename='images/model_labeled.png')


def train_autoencoder(all_ds, ds_labeled, y_labeled):
    batch_size_unlabeled = 256

    autoencoder, encoder = create_autoencoder(all_ds[0].shape)
    autoencoder.compile(optimizer=Adam(), loss='mse')

    if not on_server:
        plot_model(autoencoder, to_file='images/_model_autoencoder.png', show_shapes=True)
        Image(filename='images/_model_autoencoder.png')

    # TRAINING (se i parametri sono stati già salvati, li si prende da file system)
    model_loaded = False

    name_file_model = 'parameters/' + dataset_name + '_duplex_pretraining_' + ('conv' if use_convolutional else 'fwd')
    mean_value = np.mean(all_ds)

    try:
        autoencoder.load_weights(name_file_model)
        model_loaded = True
    except Exception:
        pass

    if not model_loaded:
        use_second_method = True

        print("Training autoencoder...")
        print("Mean value:", mean_value)

        if use_second_method and len(ds_labeled) > 0:
            # si usano due loss
            encoder.compile(optimizer=Adam(), loss=[custom_layers.get_my_pretraining_loss()], loss_weights=[gamma_ce])

            # esempi organizzati per classi
            samples_per_class = []
            for c in positive_classes:
                ds_class, y_class = get_data.filter_ds(ds_labeled, y_labeled, [c])
                samples_per_class.append((ds_class, y_class))

            for epoch in range(autoencoder_n_epochs):
                print("EPOCH:", epoch, "/", autoencoder_n_epochs)

                autoencoder.fit(all_ds, all_ds, batch_size=batch_size_unlabeled, epochs=1, shuffle=True)
                while False:
                    index_unlabeled = 0
                    # unlabeled train on batch
                    if (index_unlabeled + 1) * batch_size_unlabeled > all_ds.shape[0]:
                        loss = autoencoder.train_on_batch(x=all_ds[index_unlabeled * batch_size_unlabeled::],
                                                          y=all_ds[index_unlabeled * batch_size_unlabeled::])

                        print("loss U:", np.round(loss, 5))
                        break # next epoch
                    else:
                        loss = autoencoder.train_on_batch(
                            x=all_ds[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled],
                            y=all_ds[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled])
                        index_unlabeled += 1

                # supervised loss
                for class_samples, y_class in samples_per_class:
                    loss = encoder.train_on_batch(x=[class_samples], y=[y_class])
                    print("loss L:", np.round(loss, 5))

                if epoch % 20 == 0:
                    gc.collect()

        else:
            autoencoder.fit(all_ds, all_ds, batch_size=batch_size_unlabeled, epochs=autoencoder_n_epochs, shuffle=True)

        autoencoder.save_weights(name_file_model + ".h5")

    # show dataset
    if False:
        for i in range(36):
            ax1 = plt.subplot(6, 6, 1 + i)
            # plt.subplot(660+1+i)

            if i % 2 == 0:
                ax1.imshow(all_ds[i])
            else:
                rec = autoencoder.predict(all_ds[i - 1:i])[0]
                ax1.imshow(rec)
        plt.show()

    # calcolo loss
    ds_pred = autoencoder.predict(all_ds)
    loss = tf.keras.losses.mean_squared_error(all_ds, ds_pred)
    loss = np.mean(loss.numpy())

    print("MSE LOSS FOR AUTOENCODER AFTER PRETRAINING:", loss, ", normalized:", loss / (mean_value ** 2))

    return autoencoder, encoder


def run_duplex(model_unlabeled, model_labeled, encoder, clustering_layer,
               ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, kld_weight=0.1, ce_weight=0.1,
               mse_weight=1, maxiter=10000, upd_interval=140):

    batch_size_unlabeled = 256
    miniter = 200
    tol = 0.001  # tolerance threshold to stop training
    do_kld = kld_weight > 0

    print("Beginning training, maxiter:", maxiter, ", tol:", tol, ", ce_weight:", ce_weight, ", kld_weight:", kld_weight)

    y_pred_last = None
    if len(ds_labeled) > 0:
        all_x = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
        all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)
    else:
        all_x = ds_unlabeled
        all_y = y_unlabeled

    index_labeled_for_plot = np.array([i < len(ds_labeled) for i, _ in enumerate(all_x)])

    # ci si assicura un equo processamento di esempi etichettati e non
    labeled_interval = max(1, int(((1 / perc_labeled) - 1) * (batch_size_labeled / batch_size_unlabeled))) if len(ds_labeled) > 0 and gamma_ce > 0 else -1
    plot_interval = int(len(all_x) / batch_size_unlabeled) * (4 if dataset_name == "reuters" else 16)
    measures_interval = upd_interval * (10 if dataset_name == "reuters" else 1)

    print("update_interval:", upd_interval, ", batch_size_unlabeled:", batch_size_unlabeled,
          ", labeled_interval:", labeled_interval, ", batch_size_labeled:", batch_size_labeled,
          ", plot_interval:", plot_interval, ", measures_interval:", measures_interval)

    # compile models
    #sup_loss = custom_layers.get_my_argmax_loss(batch_size_labeled, y_prod_type=ce_function_type, m_prod_type=m_prod_type, num_classes=num_classes)
    sup_loss = custom_layers.get_my_gravity_loss(batch_size_labeled, y_prod_type=ce_function_type, m_prod_type=m_prod_type, num_classes=num_classes)

    if which_optimizer == "sgd":
        model_unlabeled.compile(loss=['kld', 'mse'], loss_weights=[kld_weight, mse_weight], optimizer=SGD())
        model_labeled.compile(loss=[sup_loss], loss_weights=[ce_weight], optimizer=SGD())
    else:
        model_unlabeled.compile(loss=['kld', 'mse'], loss_weights=[kld_weight, mse_weight], optimizer=Adam())
        model_labeled.compile(loss=[sup_loss], loss_weights=[ce_weight], optimizer=Adam())

    # bisogna avere anche le etichette per i negativi (tutte impostate a zero)
    num_clusters = len(clustering_layer.get_centroids())

    if labeled_interval != -1:
        temp_y_for_model_labeled = keras.utils.to_categorical(y_labeled)
        temp_y_for_model_labeled = temp_y_for_model_labeled[:, positive_classes]

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
        if show_plots and ite % plot_interval == 0:
            y_pred_p, _ = model_unlabeled.predict(all_x, verbose=0)
            y_pred_p = y_pred_p.argmax(1)

            if do_kld or (ite % plot_interval * 5) != 0 or supervised_loss_type != "on_encoded":
                centroids = clustering_layer.get_centroids()
            else:
                # si plottano i centroidi ricalcolati
                centroids = get_centroids(all_x, ds_unlabeled, ds_labeled, y_labeled, encoder)

            plot_2d(encoder.predict(all_x), y_pred_p, all_y, index_labeled_for_plot, centroids, perc_to_compute=0.6 if ite == 0 else 0.2)
            del y_pred_p

        if labeled_interval != -1 and ite % int(len(ds_labeled) / batch_size_labeled) == 0:
            # shuffle data (experimental)
            shuffler1 = np.random.permutation(len(ds_labeled))
            ds_labeled = ds_labeled[shuffler1]
            y_labeled = y_labeled[shuffler1]
            y_for_model_labeled = y_for_model_labeled[shuffler1]

            del shuffler1

        # labeled training (questo metodo funziona fintantoche gli esempi labeled sono meno degli unlabeled)
        if labeled_interval != -1 and ite % labeled_interval == 0:
            if (index_labeled + 1) * batch_size_labeled > ds_labeled.shape[0]:
                index_labeled = 0

            loss = model_labeled.train_on_batch(
                x=ds_labeled[index_labeled * batch_size_labeled:(index_labeled + 1) * batch_size_labeled],
                y=[y_for_model_labeled[index_labeled * batch_size_labeled:(index_labeled + 1) * batch_size_labeled]
                   ])
            index_labeled += 1

            print('Ite:', "{:4.0f}".format(ite), ", L loss:", loss)

        # update target probability
        if ite % upd_interval == 0:
            #if not do_kld and ite > 0 and supervised_loss_type == "on_encoded" and (ite % (measures_interval * 1)) == 0:
                # reinizializzazione centroidi se non si sta facendo la kld (per le performances)
            #    clustering_layer.set_weights([get_centroids(all_x, ds_unlabeled, ds_labeled, y_labeled, encoder)])

            # PREDICT
            q, _ = model_unlabeled.predict(all_x, verbose=0)
            p = custom_layers.target_distribution(q)  # update the auxiliary target distribution p
            y_pred_u = q.argmax(1)

            if ite % measures_interval == 0:
                # evaluate the clustering performance
                custom_layers.print_measures(all_y, y_pred_u, classes, ite=ite)

            # check stop criterion
            delta_label = np.sum(y_pred_u != y_pred_last).astype(np.float32) / y_pred_u.shape[0] if y_pred_last is not None else 1
            y_pred_last = y_pred_u
            if ite > miniter and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                'Reached tolerance threshold. Stopping training.'
                break

            del y_pred_u, q, _

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

        if ite % 200 == 0:
            print('Ite:', "{:4.0f}".format(ite), ", U loss:", loss)

        if ite % 2000 == 0:
            gc.collect()


def run_duplex_second(model_unlabeled, model_labeled, encoder, clustering_layer,
               ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, kld_weight=0.1, ce_weight=0.1,
               mse_weight=1, maxiter=10000, upd_interval=140):

    batch_size_unlabeled = 256
    miniter = 200
    tol = 0.001  # tolerance threshold to stop training

    print("Beginning training, maxiter:", maxiter, ", tol:", tol, ", ce_weight:", ce_weight, ", kld_weight:", kld_weight)

    y_pred_last = None
    if len(ds_labeled) > 0:
        all_x = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
        all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)
    else:
        all_x = ds_unlabeled
        all_y = y_unlabeled

    index_labeled_for_plot = np.array([i < len(ds_labeled) for i, _ in enumerate(all_x)])

    # ci si assicura un equo processamento di esempi etichettati e non
    labeled_interval = 1 if len(ds_labeled) > 0 and gamma_ce > 0 else -1
    plot_interval = 4 if dataset_name == "reuters" else 16
    measures_interval = upd_interval * (10 if dataset_name == "reuters" else 1)

    print("update_interval:", upd_interval, ", batch_size_unlabeled:", batch_size_unlabeled,
          ", labeled_interval:", labeled_interval, ", batch_size_labeled:", batch_size_labeled,
          ", plot_interval:", plot_interval, ", measures_interval:", measures_interval)

    # compile models
    sup_loss = custom_layers.get_my_pretraining_loss()

    if which_optimizer == "sgd":
        model_unlabeled.compile(loss=['kld', 'mse'], loss_weights=[kld_weight, mse_weight], optimizer=SGD())
        model_labeled.compile(loss=[sup_loss], loss_weights=[ce_weight], optimizer=SGD())
    else:
        model_unlabeled.compile(loss=['kld', 'mse'], loss_weights=[kld_weight, mse_weight], optimizer=Adam())
        model_labeled.compile(loss=[sup_loss], loss_weights=[ce_weight], optimizer=Adam())

    # esempi organizzati per classi
    samples_per_class = []
    for c in positive_classes:
        ds_class, y_class = get_data.filter_ds(ds_labeled, y_labeled, [c])
        samples_per_class.append((ds_class, y_class))
    samples_per_class = np.array(samples_per_class)

    iter_per_epoch = int(len(all_x) / batch_size_unlabeled)
    ite = 0
    index_unlabeled = 0

    for epoch in range(int(maxiter / iter_per_epoch)):
        print("EPOCH:", epoch)

        # show data each epoch
        if show_plots and epoch % plot_interval == 0:
            y_pred_p, _ = model_unlabeled.predict(all_x, verbose=0)
            y_pred_p = y_pred_p.argmax(1)
            centroids = clustering_layer.get_centroids()

            # plot
            perc_to_compute = 0.7 if epoch == 0 else 0.2
            plot_2d(encoder.predict(all_x), y_pred_p, all_y, index_labeled_for_plot, centroids, perc_to_compute=perc_to_compute)
            del y_pred_p

        # labeled training
        if labeled_interval != -1:
            # shuffle data
            shuffler1 = np.random.permutation(len(samples_per_class))
            samples_per_class = samples_per_class[shuffler1]

            del shuffler1

            # supervised loss
            for class_samples, y_class in samples_per_class:
                loss = model_labeled.train_on_batch(x=[class_samples], y=[y_class])
                print("loss L:", np.round(loss, 5))

        # unlabeled training
        while True:
            # update target probability
            if ite % upd_interval == 0:
                # PREDICT
                q, _ = model_unlabeled.predict(all_x, verbose=0)
                p = custom_layers.target_distribution(q)  # update the auxiliary target distribution p
                y_pred_u = q.argmax(1)

                # evaluate the clustering performance
                if ite % measures_interval == 0:
                    custom_layers.print_measures(all_y, y_pred_u, classes, ite=ite)

                # check stop criterion
                delta_label = np.sum(y_pred_u != y_pred_last).astype(np.float32) / y_pred_u.shape[0] if y_pred_last is not None else 1
                y_pred_last = y_pred_u
                if ite > miniter and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    'Reached tolerance threshold. Stopping training.'
                    break

                del y_pred_u, q, _

            # unlabeled train on batch
            if (index_unlabeled + 1) * batch_size_unlabeled > all_x.shape[0]:
                loss = model_unlabeled.train_on_batch(x=all_x[index_unlabeled * batch_size_unlabeled::],
                                                      y=[p[index_unlabeled * batch_size_unlabeled::],
                                                         all_x[index_unlabeled * batch_size_unlabeled::]])
                index_unlabeled = 0
                print('Iter:', ite, ", U loss:", loss)
            else:
                loss = model_unlabeled.train_on_batch(
                    x=all_x[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled],
                    y=[p[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled],
                       all_x[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled]])
                index_unlabeled += 1

            ite += 1
            if ite % 2000 == 0:
                gc.collect()

            if index_unlabeled == 0:
                break # fine epoca


def main():

    # parametri calcolati
    global classes, num_classes, num_pos_classes, negative_classes, positive_classes

    classes = positive_classes.copy()
    classes.extend(negative_classes)

    num_classes = len(classes)
    num_pos_classes = len(positive_classes)

    global ce_function_type  # todo

    # print dei parametri
    print("\n\n ------------------------------------------- ")
    print("Showing plots:", show_plots)
    print("supervised_loss_type:", supervised_loss_type, ", centroid_init:", centroid_init, ", ce_function_type:", ce_function_type, ", m_prod_type:", m_prod_type, ", gamma_ce:", gamma_ce, ", gamma_kld:", gamma_kld,
          ", update_interval:", update_interval, ", batch_size_labeled:", batch_size_labeled, ", skip_supervised_pretraining:", skip_supervised_pretraining)
    print("perc_labeled:", perc_labeled, ", dataset_name:", dataset_name, ", use_convolutional:", use_convolutional,  ", optimizer:" , which_optimizer, ", perc_ds:", perc_ds)
    print("positive_classes", positive_classes, "\nnegative_classes", negative_classes)

    # dataset
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = get_dataset()

    if len(ds_labeled) > 0:
        all_ds = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
        all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)
    else:
        all_ds = ds_unlabeled
        all_y = y_unlabeled
    index_labeled_for_plot = np.array([i < len(ds_labeled) for i, _ in enumerate(all_ds)])

    # PRETRAINING autoencoder
    autoencoder, encoder = train_autoencoder(all_ds, ds_labeled, y_labeled)

    # INIZIO PRETRAINING SUPERVISIONATO
    if not skip_supervised_pretraining and len(ds_labeled) > 0:
        centroids = get_centroids(all_ds, ds_unlabeled, ds_labeled, y_labeled, encoder)

        # models
        model_unlabeled, model_labeled, clustering_layer = init_models(centroids, encoder, autoencoder)
        plot_models(model_unlabeled, model_labeled)

        # train
        #model_unlabeled.load_weights("parameters/" + dataset_name + "_duplex_pretraining2_unlabeled")
        #model_labeled.load_weights("parameters/" + dataset_name + "_duplex_pretraining2_labeled")

        # l'intervallo di update serve solo per calcolare il delta degli elementi cambiati di classe
        # per velocizzare l'esecuzione è meglio incrementarlo

        run_duplex(model_unlabeled, model_labeled, encoder, clustering_layer, ds_labeled, y_labeled, ds_unlabeled,
                   y_unlabeled, kld_weight=0, ce_weight=gamma_ce,
                   upd_interval=update_interval * 7, maxiter=6000)

        model_unlabeled.save_weights("parameters/" + dataset_name + "_duplex_pretraining2_unlabeled")
        model_labeled.save_weights("parameters/" + dataset_name + "_duplex_pretraining2_labeled")

    # FINE ALLENAMENTO SUP

    # CUSTOM TRAINING (tutte le classi)
    # run k means for cluster centers
    centroids = get_centroids(all_ds, ds_unlabeled, ds_labeled, y_labeled, encoder)

    # models
    model_unlabeled, model_labeled, clustering_layer = init_models(centroids, encoder, autoencoder)
    plot_models(model_unlabeled, model_labeled)

    # fit
    if True:
        run_duplex_second(model_unlabeled, model_labeled, encoder, clustering_layer, ds_labeled, y_labeled,
                   ds_unlabeled, y_unlabeled, kld_weight=gamma_kld, ce_weight=gamma_ce, upd_interval=update_interval)
        #run_duplex(model_unlabeled, model_labeled, encoder, clustering_layer, ds_labeled, y_labeled,
        #           ds_unlabeled, y_unlabeled, kld_weight=gamma_kld, ce_weight=gamma_ce * 0, upd_interval=update_interval)

        model_unlabeled.save_weights("parameters/" + dataset_name + "_duplex_trained_unlabeled")
        model_labeled.save_weights("parameters/" + dataset_name + "_duplex_trained_labeled")
    else:
        model_unlabeled.load_weights("parameters/" + dataset_name + "_duplex_trained_unlabeled")
        model_labeled.load_weights("parameters/" + dataset_name + "_duplex_trained_labeled")

    print("END OF TRAINING")
    # FINE TRAINING

    # METRICHE
    for reinit_centers in [False]:

        if reinit_centers:
            print("Re-initializing centers")
            centroids = get_centroids(all_ds, ds_unlabeled, ds_labeled, y_labeled, encoder)
            # models
            model_unlabeled, model_labeled, clustering_layer = init_models(centroids, encoder, autoencoder)

        # TRAINING DATA
        print("Test on TRAINING DATA")

        # accuratezza
        y_pred = model_unlabeled.predict(all_ds, verbose=0)[0].argmax(1)
        x_embedded_encoder = encoder.predict(all_ds)
        custom_layers.print_measures(all_y, y_pred, classes, x_for_silouhette=x_embedded_encoder)

        # plot
        centroids = clustering_layer.get_centroids()
        plot_2d(x_embedded_encoder, y_pred, all_y, index_labeled_for_plot, centroids)

        # VALIDATION DATA
        if len(x_val) > 1:
            print("Test on VALIDATION DATA")

            # accuratezza
            y_pred = model_unlabeled.predict(x_val, verbose=0)[0].argmax(1)
            x_embedded_encoder = encoder.predict(x_val)
            custom_layers.print_measures(y_val, y_pred, classes, x_for_silouhette=x_embedded_encoder)

            # plot
            plot_2d(x_embedded_encoder, y_pred, y_val, index_labeled_for_plot, centroids)


# parametri per il training
perc_labeled = 0.1
perc_ds = 1
dataset_name = 'fashion'
use_convolutional = True
which_optimizer = "adam" #sgd o adam, meglio adam

# iperparametri del modello
autoencoder_n_epochs = 100
batch_size_labeled = -1
gamma_kld = 0.1
gamma_ce = 1
skip_supervised_pretraining = True
supervised_loss_type = "on_encoded" # on_cluster o on_encoded

ce_function_type = "all" #all diff o same, meglio all
m_prod_type = "ce"

update_interval = -1
centroid_init = "kmeans" # forse meglio gm che kmeans
do_suite_test = False
show_plots = True


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

    parser.add_argument('--centroid_init')
    parser.add_argument('--do_suite_test')
    parser.add_argument('--positive_classes')
    parser.add_argument('--negative_classes')
    parser.add_argument('--show_plots')

    parser.add_argument('--skip_supervised_pretraining')
    parser.add_argument('--supervised_loss_type')
    parser.add_argument('--which_optimizer')

    args = parser.parse_args()

    global use_convolutional, perc_labeled, perc_ds, dataset_name, batch_size_labeled, gamma_kld, gamma_ce,\
        ce_function_type, m_prod_type, update_interval, do_suite_test, positive_classes, negative_classes, centroid_init,\
        show_plots, skip_supervised_pretraining, supervised_loss_type, which_optimizer

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
    else:
        batch_size_labeled = 240 if dataset_name == "reuters" else 300 if dataset_name == "ups" else 450

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
    elif update_interval == -1:
        update_interval = 3 if dataset_name == "reuters" else (30 if dataset_name == "ups" else 140)

    if args.centroid_init:
        centroid_init = args.centroid_init
    if args.supervised_loss_type:
        supervised_loss_type = args.supervised_loss_type
    if args.which_optimizer:
        which_optimizer = args.which_optimizer
    if args.do_suite_test:
        do_suite_test = args.do_suite_test == 'True'
    if args.show_plots:
        show_plots = args.show_plots == 'True'
    if args.skip_supervised_pretraining:
        skip_supervised_pretraining = args.skip_supervised_pretraining == 'True'

    if args.positive_classes:
        positive_classes = []
        for s in args.positive_classes.split(','):
            positive_classes.append(int(s))
    if args.negative_classes:
        negative_classes = []
        for s in args.negative_classes.split(','):
            negative_classes.append(int(s))

    # vincoli
    if dataset_name == "reuters":
        # 4 classi
        positive_classes = [c for c in positive_classes if c < 4]
        negative_classes = [c for c in negative_classes if c < 4]
        use_convolutional = False


# lettura parametri
read_args()
if do_suite_test:

    for ds in ["fashion", "mnist", "ups", "cifar", "reuters"]:
        dataset_name = ds

        for c in [False, True]:
            use_convolutional = c
            main()

else:
    main()

