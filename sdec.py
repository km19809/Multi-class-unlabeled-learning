import numpy as np
import matplotlib.pyplot as plt
import custom_layers, get_data
import keras
from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.manifold import TSNE
import tensorflow as tf
import datetime
import gc, os
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

plt.rcParams["figure.figsize"] = [16, 9]

# tf.compat.v1.disable_eager_execution()
print("Version rf:", tf.__version__)
print("Eager:", tf.executing_eagerly())


def read_args():
    # Si specificano i parametri in input che modificano il comportamento del sistema
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runs')
    parser.add_argument('--arg_show_plots')
    parser.add_argument('--data_preparation')
    parser.add_argument('--perc_to_show')
    parser.add_argument('--do_suite_test')
    parser.add_argument('--maxiter')

    parser.add_argument('--positive_classes')
    parser.add_argument('--negative_classes')

    parser.add_argument('--perc_labeled')
    parser.add_argument('--perc_ds')
    parser.add_argument('--dataset_name')

    parser.add_argument('--batch_size_labeled')
    parser.add_argument('--update_interval')
    parser.add_argument('--gamma_kld')
    parser.add_argument('--gamma_sup')
    parser.add_argument('--beta_sup_same')
    parser.add_argument('--beta_sup_diff')
    parser.add_argument('--embedding_dim')
    parser.add_argument('--reg_central_code')
    parser.add_argument('--gamma_sparse')
    parser.add_argument('--rho_sparse')

    parser.add_argument('--epochs_pretraining')
    parser.add_argument('--epochs_clustering')

    args = parser.parse_args()

    global num_runs, arg_show_plots, maxiter, perc_to_show, do_suite_test, arg_positive_classes, arg_negative_classes, \
        perc_labeled, perc_ds, dataset_name, arg_batch_size_labeled, arg_update_interval, gamma_kld, gamma_sup,  data_preparation, \
        beta_sup_same, beta_sup_diff, embedding_dim, epochs_pretraining, epochs_clustering, reg_central_code, gamma_sparse, rho_sparse

    if args.num_runs:
        num_runs = int(args.num_runs)
    if args.maxiter:
        maxiter = int(args.maxiter)
    if args.perc_to_show:
        perc_to_show = float(args.perc_to_show)
    if args.arg_show_plots:
        arg_show_plots = args.arg_show_plots == 'True'
    if args.do_suite_test:
        do_suite_test = args.do_suite_test == 'True'
    if args.data_preparation:
        data_preparation = args.data_preparation == 'True'

    if args.positive_classes:
        arg_positive_classes = []
        for s in args.positive_classes.split(','):
            arg_positive_classes.append(int(s))
    if args.negative_classes:
        arg_negative_classes = []
        for s in args.negative_classes.split(','):
            arg_negative_classes.append(int(s))

    if args.perc_labeled:
        perc_labeled = float(args.perc_labeled)
    if args.perc_ds:
        perc_ds = float(args.perc_ds)
    if args.dataset_name:
        dataset_name = args.dataset_name
    if args.update_interval:
        arg_update_interval = int(args.update_interval)
    if args.batch_size_labeled:
        arg_batch_size_labeled = int(args.batch_size_labeled)

    if args.gamma_kld:
        gamma_kld = float(args.gamma_kld)
    if args.gamma_sup:
        gamma_sup = float(args.gamma_sup)
    if args.beta_sup_same:
        beta_sup_same = float(args.beta_sup_same)
    if args.beta_sup_diff:
        beta_sup_diff = float(args.beta_sup_diff)
    if args.reg_central_code:
        reg_central_code = float(args.reg_central_code)
    if args.gamma_sparse:
        gamma_sparse = float(args.gamma_sparse)
    if args.rho_sparse:
        rho_sparse = float(args.rho_sparse)

    if args.epochs_pretraining:
        epochs_pretraining = int(args.epochs_pretraining)
    if args.epochs_clustering:
        epochs_clustering = int(args.epochs_clustering)
    if args.embedding_dim:
        embedding_dim = int(args.embedding_dim)


def get_dataset():

    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = \
        get_data.get_data(positive_classes, negative_classes,
            perc_labeled, flatten_data=True, perc_size=perc_ds,
            dataset_name=dataset_name,
            data_preparation=data_preparation or dataset_name == "har", print_some=False)

    global batch_size_labeled

    if arg_batch_size_labeled != -1:
        batch_size_labeled = arg_batch_size_labeled
    else:
        # calcolo automatico del batch size labeled
        best_size = None
        best_samples_left = len(ds_labeled) + 1
        size = 400

        # si sceglie la grandezza che scarta meno esempi possibile
        while size >= 250:
            samples_left = len(ds_labeled) % size
            if samples_left < best_samples_left:
                best_samples_left = samples_left
                best_size = size
            size -= 1

        batch_size_labeled = best_size

    # esigenze per la loss
    if len(ds_labeled) % batch_size_labeled != 0:

        if batch_size_labeled > len(ds_labeled):
            # caso limite
            batch_size_labeled = len(ds_labeled)
        else:
            ds_labeled = ds_labeled[:-(len(ds_labeled) % batch_size_labeled)]
            y_labeled = y_labeled[:-(len(y_labeled) % batch_size_labeled)]

    return ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val


def plot_2d(x, y_pred, y_true, index_labeled, label_image, centroids=None, perc_to_show=1., show_fig=False):

    if centroids is None:
        centroids = []

    # si prende una parte dei dati (usato per velocizzare)
    shuffler1 = np.random.permutation(len(x))
    indexes_to_take = np.array([t for i, t in enumerate(shuffler1) if i < len(shuffler1) * perc_to_show])
    x_for_tsne = x[indexes_to_take]
    labeled_for_tsne = index_labeled[indexes_to_take]

    # get data in 2D (include centroids)
    data_for_tsne = np.concatenate((x_for_tsne, centroids), axis=0) if len(centroids) else x_for_tsne

    if len(data_for_tsne[0]) == 2:
        x_embedded = data_for_tsne
    else:
        x_embedded = TSNE(n_components=2, verbose=0).fit_transform(data_for_tsne)

    vis_x = x_embedded[:-len(centroids), 0] if len(centroids) else x_embedded[:, 0]
    vis_y = x_embedded[:-len(centroids), 1] if len(centroids) else x_embedded[:, 1]

    labeled_samples_x = np.array([x for i, x in enumerate(vis_x) if labeled_for_tsne[i]])
    labeled_samples_y = np.array([x for i, x in enumerate(vis_y) if labeled_for_tsne[i]])

    # 4 figures
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Predicted vs True | All vs Labeled')
    cmap = plt.cm.get_cmap("jet", 256)

    # PREDICTED
    if y_pred is not None:

        # all
        y_pred_for_tsne = y_pred[indexes_to_take]
        ax1.scatter(vis_x, vis_y, c=y_pred_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.3)

        # labeled
        labeled_y_for_tsne = np.array([x for i, x in enumerate(y_pred_for_tsne) if labeled_for_tsne[i]])
        ax3.scatter(labeled_samples_x, labeled_samples_y, c=labeled_y_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.5)

    # TRUE

    # all
    y_true_for_tsne = y_true[indexes_to_take]
    ax2.scatter(vis_x, vis_y, c=y_true_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.3)

    # labeled
    labeled_y_true_for_tsne = np.array([x for i, x in enumerate(y_true_for_tsne) if labeled_for_tsne[i]])
    ax4.scatter(labeled_samples_x, labeled_samples_y, c=labeled_y_true_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.5)

    # CENTROIDS
    if len(centroids):
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

    path = path_for_files + label_image + ".jpg"
    plt.savefig(path)

    if not on_server:
        #print("Plotting...", path)

        play_sound()
        if show_fig:
            plt.show()

    plt.close(fig)


def plot_losses(plot_name, unlabeled_losses, labeled_losses):

    fig, (ax1, ax2) = plt.subplots(2, 1)

    index = 0
    for losses in [unlabeled_losses, labeled_losses]:
        n_values = len(losses[0][1])
        for val in range(n_values):
            x = []
            y = []
            for i in range(len(losses)):
                x.append(losses[i][0])
                y.append(losses[i][1][val])

            if index == 0:
                ax = ax1
            else:
                ax = ax2

            line, = ax.plot(x, y)

            # getting legend
            if val == 0:
                legend = "Reconstruction"
            else:
                if index == 0:
                    legend = "Clustering"
                else:
                    if val == 1:
                        legend = "Supervised"
                    else:
                        legend = "Clustering"

            line.set_label(legend)

        ax.set_title("Unlabeled" if index == 0 else "Labeled")
        ax.set_ylabel("Loss")
        ax.legend()
        index += 1

    plt.xlabel("Iteration")
    path = path_for_files + plot_name + "_losses.jpg"
    plt.savefig(path)

    plt.close(fig)


def encoding_measures(all_ds, all_y, plot_name, encoder):
    encoded_data = encoder.predict(all_ds)

    n_bins = [i for i in range(len(encoded_data[0]))]

    mean = np.mean(encoded_data, axis=0)
    std = np.std(encoded_data, axis=0)

    fig, axs = plt.subplots(2, 2)

    # main
    axs[0][0].scatter(n_bins, mean, label="O")
    axs[0][0].set_ylabel("Mean")
    axs[0][1].scatter(n_bins, std, label="O")
    axs[0][1].set_ylabel("Std")

    axs[1][0].set_ylabel("Mean")
    axs[1][1].set_ylabel("Std")
    for c in classes:
        data_c, _ = get_data.filter_ds(encoded_data, all_y, [c])

        mean = np.mean(data_c, axis=0)
        std = np.std(data_c, axis=0)

        axs[1][0].scatter(n_bins, mean, label=c, alpha=0.8)
        axs[1][1].scatter(n_bins, std, label=c, alpha=0.8)

    for sub_ax in axs:
        for ax in sub_ax:
            ax.set_xlim(right=embedding_dim + 1)
            ax.legend(loc='upper right', fontsize='xx-small')

    path = path_for_files + plot_name + "_mean_codes.jpg"
    plt.savefig(path)

    plt.close(fig)


def show_activations(x, y, autoencoder):
    layer_outputs = [layer.output for layer in autoencoder.layers]
    activation_model = keras.models.Model(inputs=autoencoder.input, outputs=layer_outputs)
    n_activations = len(layer_outputs)

    # show random images of first class
    n_images_to_show = 10

    x_first_class, _ = get_data.filter_ds(x, y, [0])

    data_to_use = np.random.permutation(x_first_class)[:n_images_to_show]
    activation_output = activation_model.predict(data_to_use)

    fig, axs = plt.subplots(n_images_to_show, n_activations)

    for i in range(n_images_to_show):
        row_ax = axs[i]

        for j in range(n_activations):
            col_ax = row_ax[j]
            col_act = activation_output[j][i]

            if len(col_act.shape) == 1:
                if int(np.sqrt(col_act.shape)) == np.sqrt(col_act.shape):
                    #continue # solo immagini quadrate
                    col_act = col_act.reshape((int(np.sqrt(col_act.shape)), int(np.sqrt(col_act.shape))))
                else:
                    col_act = col_act.reshape((1, col_act.shape[0]))

            col_ax.imshow(col_act, cmap='gray')

    path = path_for_files + "Activations.jpg"
    plt.savefig(path)
    plt.close(fig)


def create_autoencoder(input_shape, act='relu', init='glorot_uniform'):

    # DIMENSIONS
    dims = [input_shape, 500, 500, 2000, embedding_dim]
    #dims = [input_shape, 484, 484, 2025, embedding_dim]

    #if dataset_name == "pendigits":
    #    dims = [input_shape, 30, 30, 125, embedding_dim]

    print("DIMS AUTOENCODER:", dims)
    n_stacks = len(dims) - 1

    input_data = Input(shape=dims[0], name='input')
    x = input_data

    # internal layers of encoder
    for i in range(n_stacks - 1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i,
                  kernel_regularizer=keras.regularizers.l2(reg_weights))(x)

    # latent hidden layer
    activity_reg = None
    if reg_central_code > 0.:
        activity_reg = keras.regularizers.l1(reg_central_code)
    elif gamma_sparse > 0.:
        activity_reg = custom_layers.SparseActivityRegulizer(gamma_sparse, rho_sparse)

    encoded = Dense(dims[-1], activation='linear', kernel_initializer=init, name='encoder_%d' % (n_stacks - 1),
                    kernel_regularizer=keras.regularizers.l2(reg_weights))(x)

    # internal layers of decoder
    x = encoded
    for i in range(n_stacks - 1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i,
                  kernel_regularizer=keras.regularizers.l2(reg_weights))(x)

    # decoder output
    x = Dense(dims[0][0], kernel_initializer=init, name='decoder_0',
              kernel_regularizer=keras.regularizers.l2(reg_weights))(x)
    decoded = x

    autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')
    encoder_model = Model(inputs=input_data, outputs=encoded, name='encoder')

    return autoencoder_model, encoder_model


def init_models(autoencoder, encoder, include_clustering, centroids=None):

    # supervised loss
    sup_loss = custom_layers.get_my_sdec_loss(batch_size_labeled, num_classes, beta_sup_same, beta_sup_diff)

    loss_labeled = ['mse', sup_loss]
    loss_unlabeled = ['mse',]

    output_labeled = [autoencoder.output, encoder.output]
    output_unlabeled = [autoencoder.output]

    loss_weights_labeled = [1, gamma_sup]
    loss_weights_unlabeled = [1]

    if include_clustering:
        unlabeled_last_layer = custom_layers.ClusteringLayer(num_classes, weights=[centroids], name='clustering')(encoder.output)

        output_labeled.append(unlabeled_last_layer)
        output_unlabeled.append(unlabeled_last_layer)

        loss_weights_labeled.append(gamma_kld)
        loss_weights_unlabeled.append(gamma_kld)

        loss_labeled.append('kld')
        loss_unlabeled.append('kld')

    # define models
    model_unlabeled = Model(inputs=encoder.input, outputs=output_unlabeled)
    model_labeled = Model(inputs=encoder.input, outputs=output_labeled)

    # compile models
    model_unlabeled.compile(loss=loss_unlabeled, loss_weights=loss_weights_unlabeled, optimizer=Adam())
    model_labeled.compile(loss=loss_labeled, loss_weights=loss_weights_labeled, optimizer=Adam())

    return model_unlabeled, model_labeled


def plot_models(model_unlabeled, model_labeled):
    if not on_server:
        plot_model(model_unlabeled, to_file='images/_model_unlabeled.png', show_shapes=True)
        Image(filename='images/_model_unlabeled.png')
        plot_model(model_labeled, to_file='images/_model_labeled.png', show_shapes=True)
        Image(filename='images/_model_labeled.png')


def run_duplex(model_unlabeled, model_labeled, encoder,
               ds_labeled, y_labeled, ds_unlabeled, y_unlabeled,
               do_clustering, max_epochs):

    if max_epochs == 0:
        return
    if update_interval % 2 != 0:
        print("Cannot have an odd update interval")
        raise Exception()

    labeled_losses = []
    unlabeled_losses = []

    # tolerance threshold to stop training
    tol = 0.001
    batch_size_unlabeled = 256
    maxiter = max_iter if do_clustering else int(max_iter / 2)

    plot_interval = arg_plot_interval if show_plots else -1

    labeled_interval = max(1, int(((1 / perc_labeled) - 1) * (batch_size_labeled / batch_size_unlabeled))) # ogni quanto eseguire un batch di esempi etichettati

    # ottenimento ds completo
    all_x = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
    all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)
    labeled_indexes = np.array([i < len(ds_labeled) for i, _ in enumerate(all_x)])

    # bisogna avere anche le etichette per i negativi (tutte impostate a zero)
    temp_y_for_model_labeled = keras.utils.to_categorical(y_labeled)
    temp_y_for_model_labeled = temp_y_for_model_labeled[:, positive_classes]

    y_for_model_labeled = np.zeros((temp_y_for_model_labeled.shape[0], num_classes), dtype='float32')

    remaining_elements = num_classes - len(positive_classes)
    if remaining_elements > 0:
        y_for_model_labeled[:, :-remaining_elements] = temp_y_for_model_labeled
    del temp_y_for_model_labeled
    # fine codice boiler

    print("Beginning training, {} clustering; interval update: {}, measures: {}, plot: {}. Max iter: {}".format("do" if do_clustering else "no", update_interval, measures_interval, plot_interval, maxiter))
    print("Batch size unlabeled: {}, labeled: {}".format(batch_size_unlabeled, batch_size_labeled))

    y_pred_last = None
    p = None
    batch_n = 0
    epoch = 0

    #shuffler_l = np.random.permutation(len(ds_labeled))

    while do_clustering or epoch < epochs_pretraining:
        # clustering exit condition
        if do_clustering and batch_n >= maxiter:
            print("Reached maximum iterations ({})".format(maxiter))
            break

        #print("EPOCH {}, Batch n° {}".format(epoch, batch_n))

        if epoch % 50 == 0:
            gc.collect()

        # plot in 2D
        if plot_interval != -1 and epoch % plot_interval == 0:
            if do_clustering:
                centroids = model_unlabeled.get_layer('clustering').get_centroids()
                y_pred_p = model_unlabeled.predict(all_x, verbose=0)[1].argmax(1)

                plot_2d(encoder.predict(all_x), y_pred_p, all_y, labeled_indexes,
                        label_image="clu_" + str(epoch), centroids=centroids, perc_to_show=perc_to_show)

                del y_pred_p
            else:
                plot_2d(encoder.predict(all_x), None, all_y, labeled_indexes,
                        label_image="pre_" + str(epoch), perc_to_show=perc_to_show)

        # evaluate the clustering performance
        if do_clustering:
            q = model_unlabeled.predict(all_x)[1]
            y_pred_new = q.argmax(1)

            print_mes = epoch % measures_interval == 0
            custom_layers.print_measures(all_y, y_pred_new, classes, print_measures=print_mes, ite=epoch)

        # shuffle labeled dataset
        shuffler_l = np.random.permutation(len(ds_labeled))
        ds_labeled_for_batch = ds_labeled[shuffler_l]
        y_for_model_labeled_for_batch = y_for_model_labeled[shuffler_l]
        if do_clustering and p is not None:
            p_for_labeled = p[labeled_indexes][shuffler_l]

        index_unlabeled = 0
        index_labeled = 0
        ite = 0

        finish_labeled = False
        finish_unlabeled = False
        stop_for_delta = False

        while not (finish_labeled and finish_unlabeled):

            # update target probability (only for clustering)
            if do_clustering and batch_n % update_interval == 0:
                # PREDICT
                q = model_unlabeled.predict(all_x)[1]
                y_pred_new = q.argmax(1)

                # check stop criterion
                if y_pred_last is not None:
                    delta_label = sum(y_pred_new[i] != y_pred_last[i] for i in range(len(y_pred_new))) / y_pred_new.shape[0]
                    if delta_label < tol:
                        print('Reached stopping criterium, delta_label ', delta_label, '< tol ', tol)
                        stop_for_delta = True
                        break

                y_pred_last = y_pred_new

                # update the auxiliary target distribution p
                p = custom_layers.target_distribution(q)
                p_for_labeled = p[labeled_indexes][shuffler_l]
                p_for_unlabeled = p[[not v for v in labeled_indexes]]

            # unlabeled train
            if not finish_unlabeled:
                if (index_unlabeled + 1) * batch_size_unlabeled >= ds_unlabeled.shape[0]:
                    t_unlabeled = [ds_unlabeled[index_unlabeled * batch_size_unlabeled::]]
                    b_unlabeled = ds_unlabeled[index_unlabeled * batch_size_unlabeled::]

                    if do_clustering:
                        t_unlabeled.append(p_for_unlabeled[index_unlabeled * batch_size_unlabeled::])

                    finish_unlabeled = True
                else:
                    t_unlabeled = [ds_unlabeled[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled]]
                    b_unlabeled = ds_unlabeled[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled]

                    if do_clustering:
                        t_unlabeled.append(p_for_unlabeled[index_unlabeled * batch_size_unlabeled:(index_unlabeled + 1) * batch_size_unlabeled])

                    index_unlabeled += 1

                losses = model_unlabeled.train_on_batch(b_unlabeled, t_unlabeled)
                unlabeled_losses.append([batch_n, losses[1:] if isinstance(losses, list) else [losses]])

                batch_n += 1

            # labeled train
            if not finish_labeled and ite % labeled_interval == 0:
                if (index_labeled + 1) * batch_size_labeled >= ds_labeled_for_batch.shape[0]:
                    t_labeled = [ds_labeled_for_batch[index_labeled * batch_size_labeled::],
                                 y_for_model_labeled_for_batch[index_labeled * batch_size_labeled::],]

                    b_labeled = ds_labeled_for_batch[index_labeled * batch_size_labeled::]

                    if do_clustering:
                        t_labeled.append(p_for_labeled[index_labeled * batch_size_labeled::])

                    finish_labeled = True
                else:
                    t_labeled = [ds_labeled_for_batch[index_labeled * batch_size_labeled:(index_labeled + 1) * batch_size_labeled],
                                 y_for_model_labeled_for_batch[index_labeled * batch_size_labeled:(index_labeled + 1) * batch_size_labeled]]

                    b_labeled = ds_labeled_for_batch[index_labeled * batch_size_labeled:(index_labeled + 1) * batch_size_labeled]

                    if do_clustering:
                        t_labeled.append(p_for_labeled[index_labeled * batch_size_labeled:(index_labeled + 1) * batch_size_labeled])

                    index_labeled += 1

                losses = model_labeled.train_on_batch(b_labeled, t_labeled)
                labeled_losses.append([batch_n, losses[1:]])

                batch_n += 1

            ite += 1

        if stop_for_delta:
            break
        epoch += 1

    return np.array(unlabeled_losses, dtype=object), np.array(labeled_losses, dtype=object)


def set_parameters_for_dataset():

    global classes, num_classes, negative_classes, positive_classes, num_pos_classes, update_interval

    # numero effettivo di classi nel dataset
    total_n_classes = 4 if dataset_name == "reuters" else \
                      3 if dataset_name == "waveform" else \
                      6 if dataset_name == "har" else 10

    # determinazione classi positive e negative
    if arg_negative_classes or arg_negative_classes:
        negative_classes = arg_negative_classes.copy()
        positive_classes = arg_positive_classes.copy()

        # mantenimento solo delle classi effettivamente esistenti
        positive_classes = [c for c in arg_positive_classes if c < total_n_classes]
        negative_classes = [c for c in arg_negative_classes if c < total_n_classes]
    else:
        # di default la k-esima classe è negativa
        negative_classes = [total_n_classes - 1]
        positive_classes = [i for i in range(total_n_classes - 1)]

    classes = negative_classes.copy()
    classes.extend(positive_classes)

    num_classes = len(classes)
    num_pos_classes = len(positive_classes)

    # update interval
    if arg_update_interval == -1:
        if dataset_name == "reuters":
            update_interval = 4
        elif dataset_name == "semeion":
            update_interval = 20
        elif dataset_name in ["usps", "optdigits", "har", "pendigits"]:
            update_interval = 30
        elif dataset_name in ["mnist", "fashion", "cifar"]:
            update_interval = 140
        else:
            update_interval = 50
    else:
        update_interval = arg_update_interval


def single_run(current_run):
    if show_plots:
        print("Showing plots")

    # dataset
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = get_dataset()

    all_ds = np.concatenate((ds_labeled, ds_unlabeled), axis=0)
    all_y = np.concatenate((y_labeled, y_unlabeled), axis=0)
    index_labeled_for_plot = np.array([i < len(ds_labeled) for i, _ in enumerate(all_ds)])

    # Get models
    autoencoder, encoder = create_autoencoder(ds_unlabeled[0].shape)

    # SUPERVISED PRETRAINING
    if epochs_pretraining > 0 and len(ds_labeled) > 0:
        # models
        model_unlabeled, model_labeled = init_models(autoencoder, encoder, False)

        model_pars = "parameters/" + dataset_name + ".h5"

        # train & save pars
        unlabeled_losses, labeled_losses = run_duplex(model_unlabeled, model_labeled, encoder,
                                                      ds_labeled, y_labeled, ds_unlabeled,
                                                      y_unlabeled, False, epochs_pretraining)

        model_unlabeled.save_weights(model_pars)

        # MSE loss
        ds_pred = autoencoder.predict(all_ds)
        loss = tf.keras.losses.mean_squared_error(all_ds, ds_pred)
        loss = np.mean(loss.numpy())
        print("MSE loss for autoencoder:", loss)

        if current_run == 0:
            plot_losses("Pretraining", unlabeled_losses, labeled_losses)
            encoding_measures(all_ds, all_y, "Pretraining", encoder)
            show_activations(all_ds, all_y, autoencoder)

        for c in classes:
            x_class, _ = get_data.filter_ds(all_ds, all_y, [c])
            x_class = encoder.predict(x_class)

            mean = np.mean(x_class, axis=0)
            std = np.std(x_class, axis=0)

            print("Class {} MEAN: {}".format(c, mean))
            print("Class {} std: {}".format(c, std))
            print("Mean -> mean {}; std {}".format(np.mean(mean), np.std(mean)))
            print("Std  -> mean {}; var {}".format(np.mean(std), np.var(std)))

        print("END PRETRAINING")

    # FULL TRAINING

    # kmeans for centroids
    centroids = custom_layers.get_centroids_from_kmeans(num_classes, positive_classes, ds_unlabeled, ds_labeled,
                                                        y_labeled, encoder)

    # models
    model_unlabeled, model_labeled = init_models(autoencoder, encoder, True, centroids)
    plot_models(model_unlabeled, model_labeled)

    # train
    unlabeled_losses, labeled_losses = run_duplex(model_unlabeled, model_labeled, encoder,
                                                  ds_labeled, y_labeled, ds_unlabeled,
                                                  y_unlabeled, True, epochs_clustering)

    if current_run == 0:
        plot_losses("Clustering", unlabeled_losses, labeled_losses)
        encoding_measures(all_ds, all_y, "Clustering", encoder)

    print("END OF TRAINING")

    # METRICHE

    # TRAINING DATA
    print("Test on TRAINING DATA")

    # accuratezza
    y_pred = model_unlabeled.predict(all_ds, verbose=0)[1].argmax(1)
    x_embedded_encoder = encoder.predict(all_ds)
    centroids = model_unlabeled.get_layer('clustering').get_centroids()

    if show_plots:
        plot_2d(x_embedded_encoder, y_pred, all_y, index_labeled_for_plot, "train", centroids)

    train_mes = custom_layers.print_measures(all_y, y_pred, classes, x_for_silouhette=x_embedded_encoder)

    # VALIDATION DATA
    test_mes = []
    if len(x_val) > 0:
        print("Test on VALIDATION DATA")

        # accuratezza
        y_pred = model_unlabeled.predict(x_val, verbose=0)[1].argmax(1)
        x_embedded_encoder = encoder.predict(x_val)

        if show_plots:
            plot_2d(x_embedded_encoder, y_pred, y_val, index_labeled_for_plot, "test", centroids)

        test_mes = custom_layers.print_measures(y_val, y_pred, classes, x_for_silouhette=x_embedded_encoder)

    return train_mes, test_mes


def main():
    # parametri calcolati
    set_parameters_for_dataset()

    global path_for_files, show_plots
    if not os.path.exists("logs/" + sub_path):
        os.mkdir("logs/" + sub_path)

    path_for_files = "logs/" + sub_path + dataset_name + "_" + datetime.datetime.now().strftime("%m_%d_%H_%M") + "/"
    if not os.path.exists(path_for_files):
        os.mkdir(path_for_files)

    # print dei parametri
    print("\n\n\n\n ------------------------------------------- ")
    print("Saving on {}".format(path_for_files))
    print("Dataset: {}{}, {}% labeled"\
          .format(dataset_name, "" if perc_ds == 1 else ('(' + str(perc_ds * 100) + '%)'), int(perc_labeled * 100)))
    print("Positive:", positive_classes, "\nNegative:", negative_classes)
    print("Epochs pretraining: {}, clustering: {}".format(epochs_pretraining, epochs_clustering))
    print("Gamma clustering: {}, supervised: {}; Beta same: {}, diff: {}"
          .format(gamma_kld, gamma_sup, beta_sup_same, beta_sup_diff))
    #print("Central regularization: {}; Sparse rho:{}, gamma:{}" \
    #      .format(reg_central_code, rho_sparse, gamma_sparse))

    train_tot_mes = None
    test_tot_mes = None
    for run in range(num_runs):
        print("\nRun {} of {}".format(run + 1, num_runs))
        np.random.seed(run)

        show_plots = arg_show_plots and run == 0 # plot solo al primo run

        # RUN
        train_mes, test_mes = single_run(run)

        if run == 0:
            train_tot_mes = np.array([train_mes,])
            test_tot_mes = np.array([test_mes,])
        else:
            train_tot_mes = np.concatenate((train_tot_mes, [train_mes]), axis=0)
            test_tot_mes = np.concatenate((test_tot_mes, [test_mes]), axis=0)

    # saving measures
    with open(path_for_files + "measures.log", 'w') as file_measures:
        index = 0
        file_measures.write("\t\tAcc\t\tNMI\t\tPurity\n")
        for measures in [train_tot_mes, test_tot_mes]:
            file_measures.write("MEASURES " + ("TRAIN" if index == 0 else "TEST") + "\n")

            for row_measure in measures:
                file_measures.write("\t\t")
                for measure in row_measure:
                    file_measures.write("{:6.4f}\t".format(measure))
                file_measures.write("\n")

            file_measures.write("Mean\t")
            for measure in measures.mean(axis=0):
                file_measures.write("{:6.4f}\t".format(measure))
            file_measures.write("\n")

            file_measures.write("Std \t")
            for measure in measures.std(axis=0):
                file_measures.write("{:6.4f}\t".format(measure))
            file_measures.write("\n")

            index += 1
            file_measures.write("\n")


# classi del problema
arg_positive_classes = []
arg_negative_classes = []

classes = []
num_classes = 0
num_pos_classes = 0


do_suite_test = True
num_runs = 1
arg_show_plots = False
perc_to_show = 0.3
path_for_files = ""
sub_path = ""
arg_plot_interval = 100
measures_interval = 1


# parametri per il training
perc_ds = 1
perc_labeled = 0.5
dataset_name = 'waveform'
data_preparation = False

# iperparametri del modello
arg_update_interval = -1
update_interval = -1
arg_batch_size_labeled = -1
batch_size_labeled = -1

gamma_kld = 0.1
gamma_sup = 0.1
embedding_dim = 40
beta_sup_same = 40
beta_sup_diff = 40

reg_weights = 0
reg_central_code = 0.00000
gamma_sparse = 0.00000
rho_sparse = 0.00

epochs_pretraining = 150
epochs_clustering = 200
max_iter = 5

# READING ARGUMENTS
read_args()
if do_suite_test:
    print("-------- TEST SUITE --------")

    sub_path = "waveform_logs/"

    nums = [10, 15, 25, 5]
    for i in nums:
        for j in nums:
            for k in nums:
                embedding_dim = i
                beta_sup_same = j
                beta_sup_diff = k

                main()


    #for ds in ["pendigits", "semeion", "optdigits", "har", "usps", "waveform"]:
    #    dataset_name = ds
    #    main()

else:
    main()

