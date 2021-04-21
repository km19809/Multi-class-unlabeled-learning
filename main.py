import os
import time
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
time.sleep(2)
np.random.seed(0)
format_acc = "{:5.3f}"

if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--classifier')
    parser.add_argument('--data_prep')
    parser.add_argument('--n_runs')
    parser.add_argument('--num_neg_classes')
    parser.add_argument('--validation_hyp')
    parser.add_argument('--test_suite')
    args = parser.parse_args()

    # setting dei parametri
    n_runs = 5
    perc_ds = 1
    perc_labeled = 0.5
    data_preparation = 'z_norm'
    nums_neg_classes = [1]
    validation_hyp = True

    datasets = ["landsat", "sonar", "semeion", "optdigits", "pendigits", "har", "usps", "mnist", "fashion", "waveform", "reuters"]
    classifiers = ['linearSVM', 'rbfSVM', "sdec", 'area', 'urea', "mpu"]

    if args.test_suite == "multiple_negative":
        datasets = ["mnist", "usps", "semeion"] # solo alcuni dataset
        nums_neg_classes = [2, 3]
        n_runs = 10

    elif args.test_suite == "contrastive":
        # comparazione tra sdec con loss mia e loss contrastive
        classifiers = ["sdec", "sdec_contrastive"]
        validation_hyp = "margin_test"

    elif args.test_suite == "debug":
        # test di debug
        classifiers = ["mpu"]

    if args.n_runs:
        n_runs = int(n_runs)
    if args.dataset:
        datasets = [args.dataset]
    if args.classifier:
        classifiers = [args.classifier]
    if args.data_prep:
        data_preparation = args.data_prep
    if args.num_neg_classes:
        nums_neg_classes = [args.num_neg_classes]
    if args.validation_hyp:
        validation_hyp = args.validation_hyp
    # end arguments parsing

    # print info
    print("Classifiers:", classifiers)
    print("Datasets:", datasets)
    print("Data prep:", data_preparation)
    print("Hyperparameters validation:", validation_hyp)
    print("Negative class mode:", nums_neg_classes)

    print("Perc. labeled:", perc_labeled, ", total:", perc_ds)
    print("Number of Runs:", n_runs)
    print()
    #for dataset in datasets:
    #    ds.get_dataset_info(dataset)

    # start execution
    for num_neg_classes in nums_neg_classes:

        print("\n\nNEG CLASS MODE:", num_neg_classes)

        prefix_path = datetime.datetime.now().strftime("%m_%d_%H") + "_" + str(num_neg_classes) + "_"

        total_test_accuracies = []
        for dataset_name in datasets:
            for name in classifiers:

                # get model
                if name == "linearSVM":
                    model = LinearSVM(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp)
                elif name == "rbfSVM":
                    model = RbfSVM(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp)
                elif name == "area":
                    model = AREA(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp)
                elif name == "urea":
                    model = UREA(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp)
                elif name == "mpu":
                    model = MPU(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp)
                elif name == "sdec" or name == "sdec_contrastive":
                    model = SDEC(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path, num_neg_classes, validation_hyp)

                # get test accuracies
                test_accuracies, train_accuracies = model.run_experiments()
                total_test_accuracies.append(test_accuracies)

        # print results
        print("\n\n --- RESULTS ---")

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
                curr_test_acc = total_test_accuracies[index]
                index += 1

                print(format_acc.format(np.mean(curr_test_acc)) + "±" + format_acc.format(np.std(curr_test_acc)) + "\t\t", end='')

            print()

        print("---------------")
        # fine risultati



