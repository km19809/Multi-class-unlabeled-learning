from abc import ABC, abstractmethod
import datasets
from pylab import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os, time
import pickle
from pprint import pprint
from sklearn.manifold import TSNE
plt.rcParams["figure.figsize"] = [16, 9]

random_neg_class = dict()

class BaseClassifier(ABC):

    def __init__(self, classifier_name, dataset_name, perc_ds=1, perc_labeled=0.5, data_preparation=None, n_runs=5,
                 prefix_path='', negative_class_mode="last", validate_hyp=False, negative_classes=None):
        self.num_runs = n_runs
        self.perc_ds = perc_ds
        self.dataset_name = dataset_name
        self.data_preparation = data_preparation
        self.classifier_name = classifier_name
        self.negative_class_mode = negative_class_mode
        self.validate_hyp = validate_hyp

        self.path_for_files = "logs/" + prefix_path + classifier_name
        if not os.path.exists(self.path_for_files):
            os.mkdir(self.path_for_files)

        self.path_for_files += "/" + dataset_name + "/"
        if not os.path.exists(self.path_for_files):
            os.mkdir(self.path_for_files)

        self.num_folds = 5

        # numero effettivo di classi nel dataset
        total_n_classes = datasets.get_n_classes(self.dataset_name)

        self.full_labeled = self.classifier_name in ["linearSVM", 'rbfSVM']
        self.perc_labeled = 1 if self.full_labeled else perc_labeled
        self.total_classes = total_n_classes

        # classi
        if self.full_labeled:
            self.negative_classes = []
            self.true_negative_classes = []
            self.positive_classes = [i for i in range(total_n_classes)]
        elif negative_classes is not None:
            self.negative_classes = negative_classes.copy()
            self.true_negative_classes = self.negative_classes.copy()
            self.positive_classes = [i for i in range(total_n_classes) if i not in negative_classes]
        else:
            if negative_class_mode == "last":
                # la k-esima classe è negativa
                self.negative_classes = [total_n_classes - 1]
                self.true_negative_classes = self.negative_classes.copy()
                self.positive_classes = [i for i in range(total_n_classes - 1)]
            elif negative_class_mode == "two_in_one":
                # le ultime due classi negative che confluiscono in un'unica classe
                self.negative_classes = [total_n_classes - 2]
                self.true_negative_classes = [total_n_classes - 2, total_n_classes - 1]
                self.positive_classes = [i for i in range(total_n_classes - 2)]
            elif negative_class_mode == "three_in_one":
                # le ultime tre classi negative che confluiscono in un'unica classe
                self.negative_classes = [total_n_classes - 3]
                self.true_negative_classes = [total_n_classes - 3, total_n_classes - 2, total_n_classes - 1]
                self.positive_classes = [i for i in range(total_n_classes - 3)]

        self.classes = self.positive_classes.copy()
        self.classes.extend(self.negative_classes)

        # update interval (sdec)
        if dataset_name in ["reuters", "sonar"]:
            self.update_interval = 4
        elif dataset_name == "semeion":
            self.update_interval = 20
        elif dataset_name in ["usps", "optdigits", "har", "pendigits", "waveform", "landsat"]:
            self.update_interval = 30
        elif dataset_name in ["mnist", "fashion", "cifar"]:
            self.update_interval = 140
        else:
            self.update_interval = 50

        assert self.update_interval % 2 == 0

        # se non si validano gli iperparametri ci deve essere una sola configurazione
        assert validate_hyp or len(list(itertools.product(*self.get_grid_hyperparameters().values()))) == 1

    def run_experiments(self):

        train_accuracies = []
        test_accuracies = []

        start_time = time.time()

        # get and show hyperparameters
        hyp_grid = self.get_grid_hyperparameters()
        print("\n\nSAVING on '", self.path_for_files + "'")
        print()
        print("Number of hyp configurations:", len(list(itertools.product(*hyp_grid.values()))))
        pprint(hyp_grid)
        print()

        if self.validate_hyp == "margin_test":
            # gestione delle classi negative scelte a caso
            if self.dataset_name not in random_neg_class:
                random_neg_class[self.dataset_name] = np.random.choice(self.classes, self.num_runs, False)
                print("Negative classes for ds", self.dataset_name, ":", random_neg_class[self.dataset_name])

        for k in range(self.num_runs):

            print("RUN n° {} of {}".format(k + 1, self.num_runs))

            # ottenimento dataset (split in 3 parti diverse ad ogni run)
            if self.validate_hyp == "margin_test":
                # col margin test ciò che cambia è la classe negativa
                neg_class = random_neg_class[self.dataset_name][k]
                self.negative_classes = [neg_class]
                self.true_negative_classes = self.negative_classes.copy()
                self.positive_classes = [i for i in range(self.total_classes) if i != neg_class]

            ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, x_val, y_val = \
                datasets.get_data(self.positive_classes, self.true_negative_classes,
                                  self.perc_labeled, k, True, self.perc_ds,
                                  self.dataset_name, perc_val_set=0.2 if self.validate_hyp else 0,
                                  data_preparation=self.data_preparation)

            if self.full_labeled and self.negative_class_mode in ["two_in_one", "three_in_one"]:
                # le ultime 2-3 classi si accorpano
                num = 2 if self.negative_class_mode == "two_in_one" else 3
                dest_positive_class = len(self.positive_classes) - num
                source_positive_class = [dest_positive_class + i for i in range(num)]

                def set_positive_class(y):
                    for i in range(len(y)):
                        if y[i] in source_positive_class:
                            y[i] = dest_positive_class
                set_positive_class(y_labeled)
                set_positive_class(y_test)
                set_positive_class(y_val)

            # dimensione dell'input
            input_dim = ds_labeled[0].shape[0]

            if (self.classifier_name == "sdec" or self.classifier_name == "sdec_contrastive") and k == 0:
                print("--- N° batch for epochs: {}".format(int((len(ds_unlabeled) + len(ds_labeled) / 256))))

            # si prova ogni configurazione di iperparametri
            best_hyp = None
            best_model = None
            best_accuracy = None
            best_index = -1
            current_index = 0
            list_accuracies = []
            list_histories = []
            accuracy = None

            for config in list(itertools.product(*hyp_grid.values())):
                hyp = dict()
                for i in range(len(hyp_grid)):
                    hyp[list(hyp_grid.keys())[i]] = config[i]

                # inizializzazione modello
                model = self.get_model(input_dim, hyp)

                # allenamento
                result = self.train_model(model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, hyp)
                if self.classifier_name == "sdec" or self.classifier_name == "sdec_contrastive":
                    history = result[1]
                    model = result[0] # aggiornamento modello
                else:
                    history = result

                if not self.full_labeled:
                    list_histories.append(history)

                # performances sul validation set
                if self.validate_hyp:
                    y_pred_val = self.predict(model, x_val)
                    accuracy = self.get_accuracy(y_pred_val, y_val)
                    print(config, "-> val accuracy:", accuracy)
                    list_accuracies.append(accuracy)

                # mantenimento migliori performances
                if not self.validate_hyp or best_accuracy is None or accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyp = hyp
                    best_model = model
                    best_index = current_index

                current_index += 1

            # si mostra la griglia degli iperparametri
            if self.validate_hyp:
                self.plot_grid_hyps(list_accuracies, k)

            if k == 0:
                # si mostra l'andamento della loss e dell'accuratezza relativamente al miglior modello
                if not self.full_labeled:
                    best_history = list_histories[best_index]
                    self.plot_history(best_history)

                    if self.classifier_name == "sdec" or self.classifier_name == "sdec_contrastive":
                        self.plot_clusters(best_history)

            # ora si possiede la migliore configurazione di iperparametri
            if self.validate_hyp:
                print("Best Hyp:", best_hyp, " -> ", best_accuracy)

            # Allenamento completo (solo per alcuni)
            if self.full_labeled and self.validate_hyp:
                # training effettuato con train e validation set
                ds_total_train = np.concatenate((ds_labeled, x_val), axis=0)
                y_total_train = np.concatenate((y_labeled, y_val), axis=0)

                best_model = self.get_model(input_dim, best_hyp)
                self.train_model(best_model, ds_total_train, y_total_train, None, None, None, None, best_hyp)

            # TEST accuracy
            y_pred_test = self.predict(best_model, x_test)
            acc = self.get_accuracy(y_pred_test, y_test)
            print("Test accuracy:", acc)
            test_accuracies.append(acc)

            # Training accuracy
            ds_train = ds_labeled
            y_train = y_labeled
            if not self.full_labeled:
                ds_train = np.concatenate((ds_train, ds_unlabeled), axis=0)
                y_train = np.concatenate((y_train, y_unlabeled), axis=0)

            y_pred_train = self.predict(best_model, ds_train)
            train_accuracies.append(self.get_accuracy(y_pred_train, y_train))

        # end of training
        end_time = time.time()
        print("Elapsed time in sec:", int((end_time - start_time)))

        # save and return metrics
        self.save_measures(test_accuracies, train_accuracies)
        return test_accuracies, train_accuracies

    def save_measures(self, test_accuracies, train_accuracies):
        # saving measures
        with open(self.path_for_files + "measures.log", 'w') as file_measures:
            index = 0
            file_measures.write("\t\tAccuracy\n")
            for measures in [train_accuracies, test_accuracies]:
                file_measures.write("MEASURES " + ("TRAIN" if index == 0 else "TEST") + "\n")

                for row_measure in measures:
                    file_measures.write("\t\t")
                    file_measures.write("{:6.4f}\t".format(row_measure))
                    file_measures.write("\n")

                file_measures.write("Mean\t\t")
                file_measures.write("{:6.4f}\t".format(np.mean(measures, axis=0)))
                file_measures.write("\n")

                file_measures.write("Std \t\t")
                file_measures.write("{:6.4f}\t".format(np.std(measures, axis=0)))
                file_measures.write("\n")

                index += 1
                file_measures.write("\n")

    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # loss plot
        if 'loss' in history.history:
            train_loss = history.history['loss']
            line, = ax1.plot(history.epoch, train_loss)
            line.set_label("Train")
        else:
            # sdec
            train_loss = history.history['loss_rec']
            line, = ax1.plot(history.epoch, train_loss)
            line.set_label("Train - Reconstruction")

            train_loss = history.history['loss_sup']
            line, = ax1.plot(history.epoch, train_loss)
            line.set_label("Train - Supervised")

            train_loss = history.history['loss_clu']
            line, = ax1.plot(history.epoch, train_loss)
            line.set_label("Train - Clustering")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # accuracy plot
        epochs_acc = history.epoch_acc if self.classifier_name == "sdec" or self.classifier_name == "sdec_contrastive" else history.epoch

        train_accuracy = history.history['accuracy_metric']
        test_accuracy = history.history['val_accuracy_metric']

        line, = ax2.plot(epochs_acc, train_accuracy)
        line.set_label("Train")
        line, = ax2.plot(epochs_acc, test_accuracy)
        line.set_label("Test")

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        path = self.path_for_files + "Best_history.jpg"
        plt.savefig(path)
        plt.close(fig)

        # plot per sdec
        if self.classifier_name == "sdec" or self.classifier_name == "sdec_contrastive":
            fig, ax1 = plt.subplots(1, 1)

            # loss plot
            train_loss = history.history['loss_rec']
            line, = ax1.plot(history.epoch, train_loss)
            line.set_label("Train - Reconstruction")

            # si determina da che punto iniziare il plot
            train_loss = history.history['loss_sup']
            index_to_cut = 0
            while index_to_cut < min(50, len(train_loss)) and train_loss[index_to_cut] >= 10:
                index_to_cut += 1

            line, = ax1.plot(history.epoch[index_to_cut:], train_loss[index_to_cut:])
            line.set_label("Train - Supervised")

            train_loss = history.history['loss_clu']
            line, = ax1.plot(history.epoch, train_loss)
            line.set_label("Train - Clustering")

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()

            path = self.path_for_files + "Best_history_other_plot.jpg"
            plt.savefig(path)
            plt.close(fig)

    def show_plot_grid_hyps(self):
        for k in range(self.num_runs):
            path = self.path_for_files + "Hyperparameters_map_" + str(k) + ".jpg.dump"
            list_accuracies = pickle.load(open(path, 'rb'))
            self.plot_grid_hyps(list_accuracies, k, flag_save=False, flag_show=True)

    def plot_grid_hyps(self, list_accuracies, current_k, flag_save=True, flag_show=False):

        format_s = "{:5.3f}"
        format_h = "{:.2e}"
        path = self.path_for_files + "Hyperparameters_map_" + str(current_k) + ".jpg"

        # save accuracies
        if flag_save:
            with open(path + ".dump", 'wb') as file:
                pickle.dump(list_accuracies, file)

        hyps = self.get_grid_hyperparameters()

        if len(hyps) == 1:
            # 1D plot
            x_data = hyps[list(hyps.keys())[0]]

            list_accuracies = np.array(list_accuracies).reshape((len(x_data), 1))

            fig, ax = plt.subplots()
            ax.imshow(np.transpose(list_accuracies))

            ax.set_xticks(np.arange(len(x_data)))

            ax.set_xticklabels([format_h.format(x) for x in x_data])
            ax.set_xlabel(list(hyps.keys())[0])

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            for i in range(len(x_data)):
                ax.text(i, 0, format_s.format(list_accuracies[i][0]), ha="center", va="center", color="w")

            ax.set_title("Accuracy for different hyperparameters confs")
            fig.tight_layout()

            if flag_save:
                plt.savefig(path)
            if flag_show:
                plt.show()

            plt.close(fig)

        elif len(hyps) == 2:
            # 2D plot
            x_data = hyps[list(hyps.keys())[0]]
            y_data = hyps[list(hyps.keys())[1]]

            list_accuracies = np.array(list_accuracies).reshape((len(x_data), len(y_data)))

            fig, ax = plt.subplots()
            ax.imshow(np.transpose(list_accuracies))

            ax.set_xticks(np.arange(len(x_data)))
            ax.set_yticks(np.arange(len(y_data)))

            ax.set_xticklabels([format_h.format(x) for x in x_data])
            ax.set_yticklabels([format_h.format(y) for y in y_data])
            ax.set_xlabel(list(hyps.keys())[0])
            ax.set_ylabel(list(hyps.keys())[1])

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            for i in range(len(x_data)):
                for j in range(len(y_data)):
                    ax.text(i, j, format_s.format(list_accuracies[i, j]), ha="center", va="center", color="w")

            ax.set_title("Accuracy for different hyperparameters confs")
            fig.tight_layout()

            if flag_save:
                plt.savefig(path)
            if flag_show:
                plt.show()

            plt.close(fig)

        elif len(hyps) == 3:
            # 3D plot
            x_data = hyps[list(hyps.keys())[0]]
            y_data = hyps[list(hyps.keys())[1]]
            z_data = hyps[list(hyps.keys())[2]]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            colmap = cm.ScalarMappable(cmap=cm.Greens)
            colmap.set_array(list_accuracies)

            x = []
            y = []
            z = []
            for i in range(len(x_data)):
                for j in range(len(y_data)):
                    for k in range(len(z_data)):
                        x.append(i)
                        y.append(j)
                        z.append(k)

            ax.scatter(x, y, z, s=200, c=list_accuracies, cmap='Greens')
            fig.colorbar(colmap)

            ax.set_xticks(np.arange(len(x_data)))
            ax.set_yticks(np.arange(len(y_data)))
            ax.set_zticks(np.arange(len(z_data)))

            ax.set_xticklabels([format_h.format(x) for x in x_data])
            ax.set_yticklabels([format_h.format(y) for y in y_data])
            ax.set_zticklabels([format_h.format(z) for z in z_data])

            ax.set_xlabel(list(hyps.keys())[0])
            ax.set_ylabel(list(hyps.keys())[1])
            ax.set_zlabel(list(hyps.keys())[2])

            ax.set_title("Accuracy for different hyperparameters confs")
            fig.tight_layout()

            if flag_save:
                plt.savefig(path)
            if flag_show:
                plt.show()

            plt.close('all')

        else:
            print("NO GRID FOR HYPS")

    def plot_clusters(self, history):

        perc_to_show = 1

        data_plots = history.data_plot.keys()

        for components in [2]:
            for epoch in data_plots:

                centroids = history.data_plot[epoch]['centroids']
                x = history.data_plot[epoch]['x_data']
                y_true = history.data_plot[epoch]['y_data']
                y_pred = history.data_plot[epoch]['y_pred']
                index_labeled = history.data_plot[epoch]['lab_index']

                # si prende una parte dei dati (usato per velocizzare)
                shuffler1 = np.random.permutation(len(x))
                indexes_to_take = np.array([t for i, t in enumerate(shuffler1) if i < len(x) * perc_to_show])
                x_for_tsne = x[indexes_to_take]
                labeled_for_tsne = index_labeled[indexes_to_take]

                data_for_tsne = np.concatenate((x_for_tsne, centroids), axis=0)

                if components == 2:
                    # get data in 2D (include centroids)
                    if len(data_for_tsne[0]) == 2:
                        x_embedded = data_for_tsne
                    else:
                        x_embedded = TSNE(n_components=2, verbose=0).fit_transform(data_for_tsne)

                    vis_x = x_embedded[:-len(centroids), 0]
                    vis_y = x_embedded[:-len(centroids), 1]

                    labeled_samples_x = np.array([x for i, x in enumerate(vis_x) if labeled_for_tsne[i]])
                    labeled_samples_y = np.array([x for i, x in enumerate(vis_y) if labeled_for_tsne[i]])

                    # 4 figures
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                    fig.suptitle('Epoch ' + str(epoch) + ', Predicted vs True | All vs Labeled')
                    cmap = plt.cm.get_cmap("jet", 256)

                    # PREDICTED
                    if y_pred is not None:
                        # all
                        y_pred_for_tsne = y_pred[indexes_to_take]
                        ax1.scatter(vis_x, vis_y, c=y_pred_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.3)

                        # labeled
                        labeled_y_for_tsne = np.array([x for i, x in enumerate(y_pred_for_tsne) if labeled_for_tsne[i]])
                        ax3.scatter(labeled_samples_x, labeled_samples_y, c=labeled_y_for_tsne, linewidths=0.2, marker=".",
                                    cmap=cmap, alpha=0.5)

                    # TRUE

                    # all
                    y_true_for_tsne = y_true[indexes_to_take]
                    ax2.scatter(vis_x, vis_y, c=y_true_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.3)

                    # labeled
                    labeled_y_true_for_tsne = np.array([x for i, x in enumerate(y_true_for_tsne) if labeled_for_tsne[i]])
                    ax4.scatter(labeled_samples_x, labeled_samples_y, c=labeled_y_true_for_tsne, linewidths=0.2, marker=".",
                                cmap=cmap, alpha=0.5)

                    # CENTROIDS
                    if len(centroids):
                        label_color = [index for index, _ in enumerate(centroids)]
                        ax1.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1,
                                    c=label_color,
                                    edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

                        ax2.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1,
                                    c=label_color,
                                    edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

                        ax3.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1,
                                    c=label_color,
                                    edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

                        ax4.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X", alpha=1,
                                    c=label_color,
                                    edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

                    # color bar
                    norm = plt.cm.colors.Normalize(vmax=len(self.classes) - 1, vmin=0)
                    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3)
                else:
                    # get data in 3D (include centroids)
                    if len(data_for_tsne[0]) == 3:
                        x_embedded = data_for_tsne
                    else:
                        x_embedded = TSNE(n_components=3, verbose=0).fit_transform(data_for_tsne)

                    vis_x = x_embedded[:-len(centroids), 0]
                    vis_y = x_embedded[:-len(centroids), 1]
                    vis_z = x_embedded[:-len(centroids), 2]

                    labeled_samples_x = np.array([x for i, x in enumerate(vis_x) if labeled_for_tsne[i]])
                    labeled_samples_y = np.array([x for i, x in enumerate(vis_y) if labeled_for_tsne[i]])
                    labeled_samples_z = np.array([x for i, x in enumerate(vis_z) if labeled_for_tsne[i]])

                    # 4 figures
                    fig = plt.figure(figsize=(20, 20))
                    ax1 = fig.add_subplot(221, projection='3d')
                    ax2 = fig.add_subplot(222, projection='3d')
                    ax3 = fig.add_subplot(223, projection='3d')
                    ax4 = fig.add_subplot(224, projection='3d')

                    fig.suptitle('Epoch ' + str(epoch) + ', Predicted vs True | All vs Labeled')
                    cmap = plt.cm.get_cmap("jet", 256)

                    # PREDICTED
                    if y_pred is not None:
                        # all
                        y_pred_for_tsne = y_pred[indexes_to_take]
                        ax1.scatter(vis_x, vis_y, vis_z, c=y_pred_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.3)

                        # labeled
                        labeled_y_for_tsne = np.array([x for i, x in enumerate(y_pred_for_tsne) if labeled_for_tsne[i]])
                        ax3.scatter(labeled_samples_x, labeled_samples_y, labeled_samples_z, c=labeled_y_for_tsne, linewidths=0.2, marker=".",
                                    cmap=cmap, alpha=0.5)

                    # TRUE

                    # all
                    y_true_for_tsne = y_true[indexes_to_take]
                    ax2.scatter(vis_x, vis_y, vis_z, c=y_true_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.3)

                    # labeled
                    labeled_y_true_for_tsne = np.array([x for i, x in enumerate(y_true_for_tsne) if labeled_for_tsne[i]])
                    ax4.scatter(labeled_samples_x, labeled_samples_y, labeled_samples_z, c=labeled_y_true_for_tsne, linewidths=0.2, marker=".",
                                cmap=cmap, alpha=0.5)

                    # CENTROIDS
                    if len(centroids):
                        label_color = [index for index, _ in enumerate(centroids)]
                        ax1.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], x_embedded[-len(centroids):, 2],
                                    marker="X", alpha=1,
                                    c=label_color,
                                    edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

                        ax2.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], x_embedded[-len(centroids):, 2],
                                    marker="X", alpha=1,
                                    c=label_color,
                                    edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

                        ax3.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], x_embedded[-len(centroids):, 2],
                                    marker="X", alpha=1,
                                    c=label_color,
                                    edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

                        ax4.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], x_embedded[-len(centroids):, 2],
                                    marker="X", alpha=1,
                                    c=label_color,
                                    edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

                    # color bar
                    norm = plt.cm.colors.Normalize(vmax=len(self.classes) - 1, vmin=0)
                    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3)

                path = self.path_for_files + "Clusters" + str(components) + "D_" + str(epoch) + ".jpg"
                plt.savefig(path)
                plt.close(fig)

    @abstractmethod
    def accuracy_metric(self, y_true, y_pred):
        pass

    @staticmethod
    def get_accuracy(y_pred, y_true):
        return sum([1 for i, _ in enumerate(y_pred) if y_pred[i] == y_true[i]]) / len(y_pred)

    @abstractmethod
    def get_model(self, input_dim, hyp):
        pass

    @abstractmethod
    def get_grid_hyperparameters(self):
        pass

    @abstractmethod
    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):
        pass

    @abstractmethod
    def predict(self, model, x):
        pass