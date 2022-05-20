# Author: Amedeo Racanati
# Date: 07/07/2021
#
# In this script we compare different versions of the ProtoMPUL method, where the loss is composed
# differently for each ablation study
# You can pass some arguments in order to customize the experiment
# The accuracy and f1 scores are printed for each experiment

import os
import tensorflow as tf
import numpy as np
import argparse
import datetime
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
    parser.add_argument('--num_neg_classes')  # number of classes to merge in the negative class
    parser.add_argument('--validation_hyp')  # type of hyper-parameters validation/selection
    parser.add_argument('--test_suite')  # type of experiment suite
    parser.add_argument('--generate_dataset')  # whether to generate again datasets
    parser.add_argument('--ablation_type')  # which study to make
    parser.add_argument('--do_stacked_pretraining', type=bool)  # which study to make
    args = parser.parse_args()

    # set default parameters
    n_runs = 5
    perc_ds = 1
    perc_labeled = 1
    data_preparation = 'z_norm'
    do_stacked_pretraining = [True, False]
    ablation_types = [1, 2, 3, 4, 5]
    validation_hyp = False
    generate_dataset = False

    datasets = ["waveform", "reuters", "landsat", "har", "fashion", "usps", "semeion",
                "optdigits", "pendigits", "mnist", "sonar"]
    classifiers = ["protoMPUL"]

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
    if args.do_stacked_pretraining:
        do_stacked_pretraining = [args.do_stacked_pretraining]
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
    print("Stacked pretraining:", do_stacked_pretraining)
    print("Ablation types:", ablation_types)
    print("Dataset re-generation:", generate_dataset)
    if not generate_dataset:
        print("WARNING: the negative class printed in the experiments may not be the REAL negative class, "
              "due to dataset preselection")

    print("Perc. labeled:", perc_labeled, ", total:", perc_ds)
    print("Number of Runs:", n_runs)
    print()

    # start execution
    for stacked_pretraining in do_stacked_pretraining:

        print("\n\n-------------------------- PRETRAINING STACKED:", stacked_pretraining)

        # array of accuracies and f1 metrics
        total_test_accuracies = []
        total_test_f1scores = []

        # do experiment for each dataset and version of ProtoMPUL...
        for dataset_name in datasets:
            for ablation in ablation_types:

                # prefix for the folder log path
                prefix_path = datetime.datetime.now().strftime("%m_%d_%H") + "_abl" + str(ablation) + "_"

                # get model
                model = ProtoMPUL('protoMPUL', dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, 1, validation_hyp, generate_dataset)

                # set ablation type and whether to do stacked pretraining
                model.ablation_type = ablation
                model.do_stacked_pretraining = stacked_pretraining

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



