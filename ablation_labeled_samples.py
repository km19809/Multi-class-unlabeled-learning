# Author: Amedeo Racanati
# Date: 20/04/2022
#
# In this script we show the different scores when varying the number of labeled samples
# for each positive class
# You can pass some arguments in order to customize the experiment
# The accuracy and f1 scores are printed for each experiment

import os
import tensorflow as tf
import numpy as np
import argparse
import datetime

from classifiers.SVM import LinearSVM, RbfSVM
from classifiers.UREA import UREA
from classifiers.AREA import AREA
from classifiers.ProtoMPUL import ProtoMPUL

# tensorflow setup
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)
tf.random.set_seed(0)
format_acc = "{:5.3f}"

if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')  # which dataset to use
    parser.add_argument('--classifier')  # which classifier to use
    parser.add_argument('--data_prep')  # type of data preparation (z-normalization, 01 ecc...)
    parser.add_argument('--n_runs')  # number of runs for each experiment
    parser.add_argument('--validation_hyp')  # type of hyper-parameters validation/selection
    parser.add_argument('--test_suite')  # type of experiment suite
    parser.add_argument('--generate_dataset')  # whether to generate again datasets
    args = parser.parse_args()

    # set default parameters
    n_runs = 5
    perc_ds = 1
    data_preparation = 'z_norm'
    validation_hyp = True
    generate_dataset = False
    num_neg_classes = 1

    datasets = ["semeion", "pendigits"]
    classifiers = ["protoMPUL", 'area', 'urea', 'linearSVM', 'rbfSVM',]

    if args.test_suite == "debug":
        # test for debug
        pass

    if args.n_runs:
        n_runs = int(n_runs)
    if args.dataset:
        datasets = [args.dataset]
    if args.classifier:
        classifiers = [args.classifier]
    if args.data_prep:
        data_preparation = args.data_prep
    if args.validation_hyp:
        validation_hyp = args.validation_hyp
    if args.generate_dataset:
        generate_dataset = False if args.generate_dataset == "False" else True
    # end arguments parsing

    # print info
    print("Classifiers:", classifiers)
    print("Datasets:", datasets)
    print("Data prep:", data_preparation)
    print("Hyperparameters validation:", validation_hyp)
    print("Dataset re-generation:", generate_dataset)
    if not generate_dataset:
        print("WARNING: the negative class printed in the experiments may not be the REAL negative class, "
              "due to dataset preselection")

    print("Number of Runs:", n_runs)
    print()

    # start execution
    perc_labeled_samples = [0.2, 0.4, 0.6, 0.8, 1.0]  # different % of labeled samples
    for perc_labeled in perc_labeled_samples:

        print("\n\n-------------------------- PERC LABELED:", perc_labeled)

        # prefix for the folder log path
        prefix_path = datetime.datetime.now().strftime("%m_%d_%H") + "_perc_lab" + str(perc_labeled) + "_"

        # array of accuracies and f1 metrics
        total_test_accuracies = []
        total_test_f1scores = []

        # do experiment for each dataset and each classifier...
        for dataset_name in datasets:
            for name in classifiers:

                # get model
                if name == "linearSVM":
                    model = LinearSVM(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp,generate_dataset)
                elif name == "rbfSVM":
                    model = RbfSVM(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp,generate_dataset)
                elif name == "area":
                    model = AREA(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp,generate_dataset)
                elif name == "urea":
                    model = UREA(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp,generate_dataset)
                elif name == "protoMPUL":
                    model = ProtoMPUL(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp, generate_dataset)

                # get test accuracies for the model trained on a specific dataset
                test_accuracies, train_accuracies, test_f1scores, train_f1scores = model.run_experiments()
                total_test_accuracies.append(test_accuracies)
                total_test_f1scores.append(test_f1scores)
        # end experiments execution

        # print metrics results
        for metric_print in ['ACCURACY', "F1SCORE"]:
            print("\n\n --- " + metric_print + " RESULTS ---")

            # header
            print("\t\t\t\t", end='')
            for clf_name in classifiers:
                print(clf_name, end='')
                [print("\t", end='') for _ in range(4 - len(clf_name) // 4)]
            print()

            index = 0
            for dataset_name in datasets:
                print(dataset_name, end='')
                [print("\t", end='') for _ in range(4 - len(dataset_name) // 4)]

                for clf_name in classifiers:
                    curr_test_acc = total_test_accuracies[index] if metric_print == "ACCURACY" else total_test_f1scores[index]
                    index += 1

                    print(format_acc.format(np.mean(curr_test_acc)) + "±" + format_acc.format(np.std(curr_test_acc)) +
                          "\t\t", end='')

                print()

            print("---------------")
        # end printing results



