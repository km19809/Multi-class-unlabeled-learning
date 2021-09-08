# Author: Amedeo Racanati
# Date: 07/07/2021
#
# In this script we compare different methods for the "multi-positive and unlabeled learning" problem
# You can pass some arguments in order to customize the experiment
# The test accuracies are printed as result

import os
import tensorflow as tf
import numpy as np
import argparse
import datetime
from SVM import LinearSVM, RbfSVM
from UREA import UREA
from AREA import AREA
from MPU import MPU
from SDEC import SDEC
import datasets as ds

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
    parser.add_argument('--num_neg_classes')  # number of classes to merge in the negative class
    parser.add_argument('--validation_hyp')  # type of hyper-paramters validation/selection
    parser.add_argument('--test_suite')  # type of experiment suite
    parser.add_argument('--generate_dataset')  # whether to generate again datasets
    parser.add_argument('--ablation_type')  # which study to make
    args = parser.parse_args()

    # set default parameters
    n_runs = 5
    perc_ds = 1
    perc_labeled = 0.5
    data_preparation = 'z_norm'
    nums_neg_classes = [1]
    ablation_types = [None, 1, 2, 3, 4, 5]
    validation_hyp = False
    generate_dataset = True

    datasets = ["semeion", "optdigits", "pendigits", "har", "usps", "mnist", "fashion", "waveform", "reuters", "landsat", "sonar"]
    classifiers = ["sdec"]

    if args.test_suite == "debug":
        # test for debug
        #perc_ds = 0.1
        #datasets = ["sonar", 'har']
        #classifiers = ['area', 'urea']
        #nums_neg_classes = [2,3 ]
        n_runs = 1
        generate_dataset = True
        datasets = ['semeion', 'optdigits']

    if args.n_runs:
        n_runs = int(n_runs)
    if args.dataset:
        datasets = [args.dataset]
    if args.classifier:
        classifiers = [args.classifier]
    if args.data_prep:
        data_preparation = args.data_prep
    if args.num_neg_classes:
        nums_neg_classes = [int(args.num_neg_classes)]
    if args.ablation_type:
        ablation_types = [int(args.ablation_type)]
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
    print("Negative class mode:", nums_neg_classes)
    print("Ablation types:", ablation_types)
    print("Dataset re-generation:", generate_dataset)
    if not generate_dataset:
        print("WARNING: the negative class printed in the experiments may not be the REAL negative class, "
              "due to dataset preselection")

    print("Perc. labeled:", perc_labeled, ", total:", perc_ds)
    print("Number of Runs:", n_runs)
    print()
    for dataset in datasets:
        ds.get_dataset_info(dataset)

    # start execution
    for num_neg_classes in nums_neg_classes:

        print("\n\n-------------------------- NEG CLASS MODE:", num_neg_classes)

        # array of accuracies and f1 metrics
        total_test_accuracies = []
        total_test_f1scores = []

        for dataset_name in datasets:
            for ablation in ablation_types:
                # prefix for the folder log path
                prefix_path = datetime.datetime.now().strftime("%m_%d_%H") + "_ab" + str(ablation) + "_"

                # get model
                model = SDEC('sdec', dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp, generate_dataset)
                model.ablation_type = ablation

                # check for particular datasets and skip
                if (dataset_name == "sonar" and num_neg_classes > 1) or \
                    (dataset_name in ['sonar', 'waveform'] and num_neg_classes > 2):
                    total_test_accuracies.append(0)
                    total_test_f1scores.append(0)
                else:
                    # get test accuracies for the model trained on a specific dataset
                    test_accuracies, train_accuracies, test_f1scores, train_f1scores = model.run_experiments()
                    total_test_accuracies.append(test_accuracies)
                    total_test_f1scores.append(test_f1scores)

        # print metrics results
        for metric_print in ['ACCURACY', "F1SCORE"]:
            print("\n\n --- " + metric_print + " RESULTS ---")

            # header
            print("\t\t\t\t", end='')
            for abl in ablation_types:
                print(abl, end='')
                [print("\t", end='') for _ in range(4 - len(str(abl)) // 4)]
            print()

            index = 0
            for dataset_name in datasets:
                print(dataset_name, end='')
                [print("\t", end='') for _ in range(4 - len(dataset_name) // 4)]

                for abl in ablation_types:
                    curr_test_acc = total_test_accuracies[index] if metric_print == "ACCURACY" else total_test_f1scores[index]
                    index += 1

                    print(format_acc.format(np.mean(curr_test_acc)) + "±" + format_acc.format(np.std(curr_test_acc)) +
                          "\t\t", end='')

                print()

            print("---------------")
        # end printing results



