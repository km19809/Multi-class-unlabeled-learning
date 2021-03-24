from abc import ABC, abstractmethod
import numpy as np
import get_data


class BaseClassifier(ABC):

    def __init__(self, n_runs, dataset_name, perc_ds, perc_labeled, data_preparation,
                 classifier_name):
        self.num_runs = n_runs
        self.perc_labeled = perc_labeled
        self.perc_ds = perc_ds
        self.dataset_name = dataset_name
        self.data_preparation = data_preparation
        self.classifier_name = classifier_name

        self.input_dim = None
        self.n_unlabeled = None
        self.positive_class_factors = None

        # numero effettivo di classi nel dataset
        total_n_classes = 4 if self.dataset_name == "reuters" else \
            3 if self.dataset_name == "waveform" else \
                6 if self.dataset_name == "har" else 10

        if self.classifier_name == "linearSVM":
            self.negative_classes = []
            self.positive_classes = [i for i in range(total_n_classes)]
        else:
            # di default la k-esima classe è negativa
            self.negative_classes = [total_n_classes - 1]
            self.positive_classes = [i for i in range(total_n_classes - 1)]

    def train(self):
        train_tot_mes = None
        test_tot_mes = None

        for run in range(self.num_runs):
            #print("\nRun {} of {}".format(run + 1, self.num_runs))
            np.random.seed(run)

            train_mes, test_mes = self.single_run(run)

            if run == 0:
                train_tot_mes = np.array([train_mes, ])
                test_tot_mes = np.array([test_mes, ])
            else:
                train_tot_mes = np.concatenate((train_tot_mes, [train_mes]), axis=0)
                test_tot_mes = np.concatenate((test_tot_mes, [test_mes]), axis=0)

            print("TRAIN MES:", train_mes)
            print("TEST MES:", test_mes)

        return train_tot_mes, test_tot_mes

    def get_dataset(self):
        ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = \
            get_data.get_data(self.positive_classes, self.negative_classes,
                              self.perc_labeled, flatten_data=True, perc_size=self.perc_ds,
                              dataset_name=self.dataset_name, data_preparation=self.data_preparation,
                              print_some=False)

        return ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val

    @staticmethod
    def get_accuracy(y_pred, y_true):
        return sum([1 for i, _ in enumerate(y_pred) if y_pred[i] == y_true[i]]) / len(y_pred)

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def single_run(self, current_run):
        pass