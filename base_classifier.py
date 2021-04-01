from abc import ABC, abstractmethod
import datasets
from pylab import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
plt.rcParams["figure.figsize"] = [16, 9]



class BaseClassifier(ABC):

    def __init__(self, classifier_name, dataset_name, perc_ds=1, perc_labeled=0.5, data_preparation=None, n_runs=5,
                 negative_classes=None, prefix_path=''):
        self.num_runs = n_runs
        self.perc_ds = perc_ds
        self.dataset_name = dataset_name
        self.data_preparation = data_preparation
        self.classifier_name = classifier_name

        self.path_for_files = "logs/" + prefix_path + classifier_name
        if not os.path.exists(self.path_for_files):
            os.mkdir(self.path_for_files)

        self.path_for_files += "/" + dataset_name + "/"
        if not os.path.exists(self.path_for_files):
            os.mkdir(self.path_for_files)
        print("Saving on:", self.path_for_files)

        self.num_folds = 5

        # numero effettivo di classi nel dataset
        total_n_classes = datasets.get_n_classes(self.dataset_name)

        self.full_labeled = self.classifier_name in ["linearSVM", 'rbfSVM']
        self.perc_labeled = 1 if self.full_labeled else perc_labeled

        # classi
        if self.full_labeled:
            self.negative_classes = []
            self.positive_classes = [i for i in range(total_n_classes)]
        else:
            if negative_classes is not None:
                self.negative_classes = negative_classes
                self.positive_classes = [i for i in range(total_n_classes) if i not in negative_classes]
            else:
                # di default la k-esima classe è negativa
                self.negative_classes = [total_n_classes - 1]
                self.positive_classes = [i for i in range(total_n_classes - 1)]

        self.classes = self.positive_classes.copy()
        self.classes.extend(self.negative_classes)

    def run_experiments(self):

        train_accuracies = []
        test_accuracies = []

        for k in range(self.num_runs):

            # ottenimento dataset (split in 3 parti diverse ad ogni run)
            ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, x_val, y_val = \
                datasets.get_data(self.positive_classes, self.negative_classes,
                                  self.perc_labeled, k_fold=k, flatten_data=True, perc_size=self.perc_ds,
                                  dataset_name=self.dataset_name, data_preparation=self.data_preparation,
                                  print_some=False)
            input_dim = ds_labeled[0].shape[0]

            hyp_grid = self.get_grid_hyperparameters()
            n_hyp = len(hyp_grid)

            # si prova ogni configurazione di iperparametri
            best_hyp = None
            best_model = None
            best_accuracy = 0
            best_index = -1
            current_index = 0
            list_accuracies = []
            list_histories = []

            for config in list(itertools.product(*hyp_grid.values())):
                hyp = dict()
                for i in range(n_hyp):
                    hyp[list(hyp_grid.keys())[i]] = config[i]

                # inizializzazione modello
                model = self.get_model(input_dim, hyp)

                # allenamento
                history = self.train_model(model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, hyp)
                if not self.full_labeled:
                    list_histories.append(history)

                # performances sul validation set
                y_pred_val = self.predict(model, x_val)
                accuracy = self.get_accuracy(y_pred_val, y_val)
                list_accuracies.append(accuracy)

                # mantenimento migliori performances
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyp = hyp
                    best_model = model
                    best_index = current_index

                current_index += 1
                print("{:6.4f}".format(accuracy), hyp)

            if k == 0:
                # si mostra la griglia degli iperparametri
                self.plot_grid_hyps(list_accuracies)

                # si mostra l'andamento della loss e dell'accuratezza relativamente al miglior modello
                if not self.full_labeled:
                    best_history = list_histories[best_index]
                    self.plot_history(best_history)

            # ora si possiede la migliore configurazione di iperparametri
            print("Best Hyp:", best_hyp, " -> ", best_accuracy)

            # Allenamento completo (solo per alcuni)
            if self.full_labeled:
                # training effettuato con train e validation set
                ds_total_train = np.concatenate((ds_labeled, x_val), axis=0)
                y_total_train = np.concatenate((y_labeled, y_val), axis=0)

                best_model = self.get_model(input_dim, best_hyp)
                self.train_model(best_model, ds_total_train, y_total_train, None, None, None, None, best_hyp)

            # TEST accuracy
            y_pred_test = self.predict(best_model, x_test)
            test_accuracies.append(self.get_accuracy(y_pred_test, y_test))

            # Training accuracy
            ds_train = ds_labeled
            y_train = y_labeled
            if not self.full_labeled:
                ds_train = np.concatenate((ds_train, ds_unlabeled), axis=0)
                y_train = np.concatenate((y_train, y_unlabeled), axis=0)

            y_pred_train = self.predict(best_model, ds_train)
            train_accuracies.append(self.get_accuracy(y_pred_train, y_train))

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

                file_measures.write("Mean\t")
                file_measures.write("{:6.4f}\t".format(np.mean(measures, axis=0)))
                file_measures.write("\n")

                file_measures.write("Std \t")
                file_measures.write("{:6.4f}\t".format(np.std(measures, axis=0)))
                file_measures.write("\n")

                index += 1
                file_measures.write("\n")

    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # loss plot
        train_loss = history.history['loss']

        line, = ax1.plot(history.epoch, train_loss)
        line.set_label("Train")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # accuracy plot
        train_accuracy = history.history['accuracy_metric']
        test_accuracy = history.history['val_accuracy_metric']

        line, = ax2.plot(history.epoch, train_accuracy)
        line.set_label("Train")
        line, = ax2.plot(history.epoch, test_accuracy)
        line.set_label("Test")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        path = self.path_for_files + "Best_history.jpg"
        plt.savefig(path)
        plt.close(fig)

    def plot_grid_hyps(self, list_accuracies):

        format_s = "{:5.3f}"
        format_h = "{:.2e}"
        path = self.path_for_files + "Hyperparameters_map.jpg"

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

            plt.savefig(path)
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

            plt.savefig(path)
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

            plt.savefig(path)
            plt.close(fig)

        elif len(hyps) == 4:

            # 3D Plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            colmap = cm.ScalarMappable(cmap=cm.Greys)
            colmap.set_array(list_accuracies)

            X = []
            y = []
            z = []
            for conf in list(itertools.product(*hyps.values())):
                X.append(conf[0])
                y.append(conf[1])
                z.append(conf[2])

            ax.scatter(X, y, z, s=200, c=list_accuracies, cmap='Greys')
            fig.colorbar(colmap)

            ax.set_xlabel(list(hyps.keys())[0])
            ax.set_ylabel(list(hyps.keys())[1])
            ax.set_zlabel(list(hyps.keys())[2])

            ax.set_xlim(np.min(X), np.max(X), auto=True)
            ax.set_ylim(np.min(y), np.max(y))
            ax.set_zlim(np.min(z), np.max(z))

            fig.tight_layout()

            plt.show()

        else:
            print("NO GRID FOR HYPS")

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