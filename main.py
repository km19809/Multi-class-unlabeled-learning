import numpy as np
from SVM import LinearSVM, RbfSVM
from UREA import UREA
from AREA import AREA
from MPU import MPU
from SDEC import SDEC

format_acc = "{:5.3f}"


if __name__ == '__main__':

    n_runs = 4
    perc_ds = 1
    perc_labeled = 0.5

    datasets = ["semeion", "optdigits", "pendigits", "har", "waveform", "usps"]
    classifiers = ['linearSVM', 'rbfSVM', "mpu", 'area', 'urea']

    print("N RUNS:", n_runs)

    for data_preparation in ['z_norm', '01']:
        print("\n\nDATA PREPARATION:", data_preparation)

        prefix_path = data_preparation + "_"

        total_test_accuracies = []

        for dataset_name in datasets:
            for name in classifiers:

                # get model
                if name == "linearSVM":
                    model = LinearSVM(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path)
                elif name == "rbfSVM":
                    model = RbfSVM(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path)
                elif name == "area":
                    model = AREA(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path)
                elif name == "urea":
                    model = UREA(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path)
                elif name == "mpu":
                    model = MPU(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path)
                elif name == "sdec":
                    model = SDEC(name, dataset_name, perc_ds, perc_labeled, data_preparation, n_runs, prefix_path)

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


