import numpy as np
import tensorflow.compat.v1 as tf

type_y = "int8"


def get_mean_std(data, axis=(0, 1, 2)):
    # axis param denotes axes along which mean & std reductions are to be performe
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.sqrt(((data - mean) ** 2).mean(axis=axis, keepdims=True))

    return mean, std


# restituisce il dataset Mnist suddiviso in esempi etichettati e non, più il test set
def get_data(positive_classes, negative_class, perc_labeled, flatten_data=False,
             perc_size = 1, dataset_name="mnist", data_preparation=True):
    all_class = positive_classes.copy()
    all_class.extend(negative_class)

    # get dataset
    if dataset_name == "cifar":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]
    elif dataset_name == "fashion":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else: #mnist
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # filtro per classe
    (x_train, y_train) = filter_ds(x_train, y_train, all_class)
    (x_test, y_test) = filter_ds(x_test, y_test, all_class)

    # modifiche per corretta elaborazione dei dati
    dtype = 'float32'

    if flatten_data:
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        # preprocessing
        if data_preparation:
            mean, std = get_mean_std(x_train, axis=None)

            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std
    else:
        # per la convoluzionale (ogni input deve avere sempre 3 dimensioni)
        if len(x_train.shape) < 4:
            x_train = x_train.reshape((len(x_train), x_train.shape[1], x_train.shape[2], 1))
            x_test = x_test.reshape((len(x_test), x_train.shape[1], x_train.shape[2], 1))

        # preprocessing
        if data_preparation:
            mean, std = get_mean_std(x_train)

            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std

    y_train = y_train.astype(type_y)
    y_test = y_test.astype(type_y)

    # esempi positivi e negativi
    (x_train_positive, y_train_positive) = filter_ds(x_train, y_train, positive_classes)
    (x_train_negative, y_train_negative) = filter_ds(x_train, y_train, negative_class)

    tot_labeled = int(len(x_train_positive) * perc_labeled)

    # dataset che contiene gli esempi etichettati
    x_train_labeled = np.array([x for i, x in enumerate(x_train_positive) if i < tot_labeled])
    y_train_labeled = np.array([y for i, y in enumerate(y_train_positive) if i < tot_labeled])

    # esempi non etichettati (comprende gli esempi positivi e quelli negativi)
    x_train_unlabeled = np.array([x for i, x in enumerate(x_train_positive) if i >= tot_labeled])
    y_train_unlabeled = np.array([y for i, y in enumerate(y_train_positive) if i >= tot_labeled])

    x_train_unlabeled = np.append(x_train_unlabeled, x_train_negative, axis=0)
    y_train_unlabeled = np.append(y_train_unlabeled, y_train_negative, axis=0)

    # si mischiano gli esempi non etichettati per non avere serie di esempi della stessa classe negativa
    shuffler1 = np.random.permutation(len(x_train_unlabeled))
    x_train_unlabeled = x_train_unlabeled[shuffler1]
    y_train_unlabeled = y_train_unlabeled[shuffler1]

    print("Shape x data:" + str(x_train[0].shape))
    print("Shape y data:" + str(y_train[0].shape))

    x_train_labeled = x_train_labeled[:int(len(x_train_labeled) * perc_size)]
    x_train_unlabeled = x_train_unlabeled[:int(len(x_train_unlabeled) * perc_size)]
    x_train_positive = x_train_positive[:int(len(x_train_positive) * perc_size)]
    x_train_negative = x_train_negative[:int(len(x_train_negative) * perc_size)]
    y_train_unlabeled = y_train_unlabeled[:int(len(y_train_unlabeled) * perc_size)]
    y_train_labeled = y_train_labeled[:int(len(y_train_labeled) * perc_size)]
    x_train = x_train[:int(len(x_train) * perc_size)]
    x_test = x_test[:int(len(x_test) * perc_size)]
    y_test = y_test[:int(len(y_test) * perc_size)]

    print("Total: \t\t" + str(len(x_train)))
    print("Labeled: \t" + str(len(x_train_labeled)))
    print("Unlabeled: \t" + str(len(x_train_unlabeled)))
    print("Positive: \t" + str(len(x_train_positive)))
    print("Negative: \t" + str(len(x_train_negative)))

    return x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled, x_test, y_test


def filter_ds(x_ds, y_ds, classes):

    mask = [True if y in classes else False for y in y_ds]

    return (np.array([x_ds[i] for i, v in enumerate(mask) if v]),
        np.array([y_ds[i] for i, v in enumerate(mask) if v]))









