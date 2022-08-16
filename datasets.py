# This file is used for accessing to the datasets used in the experiments

import numpy as np
import tensorflow.compat.v1 as tf
import os
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# stores the random permutation for each dataset
dataset_repo = dict()


# prints some info for a given dataset
def get_dataset_info(dataset_name):
    x, y = get_dataset(dataset_name)
    print("{}: {} samples, {} classes, {} features".format(dataset_name, len(x), len(np.unique(y)), len(x[0])))


# returns x and y data
def get_dataset(dataset_name):

    x_data = None
    y_data = None

    if dataset_name == "cifar":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_data = np.concatenate((x_train, x_test), axis=0) / 255.
        y_data = np.concatenate((y_train, y_test), axis=0)
        y_data = y_data[:, 0]
    elif dataset_name == "fashion":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_data = np.concatenate((x_train, x_test), axis=0) / 255.
        y_data = np.concatenate((y_train, y_test), axis=0)
    elif dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_data = np.concatenate((x_train, x_test), axis=0) / 255.
        y_data = np.concatenate((y_train, y_test), axis=0)
    elif dataset_name == "usps":
        x_data, y_data = load_usps()
    elif dataset_name == "reuters":
        x_data, y_data = load_reuters()
    elif dataset_name == "pendigits":
        x_data, y_data = load_pendigits()
    elif dataset_name == "semeion":
        x_data, y_data = load_semeion()
    elif dataset_name == "optdigits":
        x_data, y_data = load_optdigits()
    elif dataset_name == "har":
        x_data, y_data = load_har()
    elif dataset_name == "waveform":
        x_data, y_data = load_waveform()
    elif dataset_name == "sonar":
        x_data, y_data = load_sonar()
    elif dataset_name == "landsat":
        x_data, y_data = load_landsat()

    # shuffle data always with the same permutation
    if dataset_name not in dataset_repo:
        dataset_repo[dataset_name] = np.random.permutation(len(x_data))

    shuffler = dataset_repo[dataset_name]

    x_data = np.array(x_data)[shuffler]
    y_data = np.array(y_data)[shuffler]

    return x_data, y_data


# makes several splits for a dataset and saves it on disk
def make_dataset_for_experiments(number_of_repetitions, dataset_name, positive_classes, negative_classes,
                                 flatten_data=True, perc_size=1, data_preparation=None):

    path_dataset = "data/splitted_datasets/"
    if not os.path.exists(path_dataset):
        os.mkdir(path_dataset)

    path_dataset += dataset_name + "/"
    if not os.path.exists(path_dataset):
        os.mkdir(path_dataset)

    path_dataset += 'n' + str(len(negative_classes)) + "/"
    if not os.path.exists(path_dataset):
        os.mkdir(path_dataset)

    for i in range(number_of_repetitions):
        path_dataset_repetition = path_dataset + str(i) + "/"
        if not os.path.exists(path_dataset_repetition):
            os.mkdir(path_dataset_repetition)

        x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled, x_test, y_test, x_val, y_val = \
            split_dataset_for_experiments(dataset_name, positive_classes, negative_classes, flatten_data, perc_size, data_preparation)

        # save on disk
        np.save(path_dataset_repetition + "x_train_labeled", x_train_labeled)
        np.save(path_dataset_repetition + "y_train_labeled", y_train_labeled)
        np.save(path_dataset_repetition + "x_train_unlabeled", x_train_unlabeled)
        np.save(path_dataset_repetition + "y_train_unlabeled", y_train_unlabeled)

        np.save(path_dataset_repetition + "x_test", x_test)
        np.save(path_dataset_repetition + "y_test", y_test)
        np.save(path_dataset_repetition + "x_val", x_val)
        np.save(path_dataset_repetition + "y_val", y_val)

        np.save(path_dataset_repetition + "negative_classes", np.array(negative_classes))


# returns the data for a specific dataset and repetition, taking the samples already saved on disk
def load_dataset_for_experiments(dataset_name, num_negative_classes, n_repetition):
    path = "data/splitted_datasets/" + dataset_name + '/n' + str(num_negative_classes) + "/" + str(n_repetition) + "/"

    x_train_labeled = np.load(path + 'x_train_labeled.npy')
    y_train_labeled = np.load(path + 'y_train_labeled.npy')
    x_train_unlabeled = np.load(path + 'x_train_unlabeled.npy')
    y_train_unlabeled = np.load(path + 'y_train_unlabeled.npy')

    x_test = np.load(path + 'x_test.npy')
    y_test = np.load(path + 'y_test.npy')
    x_val = np.load(path + 'x_val.npy')
    y_val = np.load(path + 'y_val.npy')

    return x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled, x_test, y_test, x_val, y_val


# split the dataset in train/validation/test set
def split_dataset_for_experiments(dataset_name, positive_classes, negative_classes, flatten_data,
             perc_size, data_preparation, print_info=False):

    all_classes = positive_classes.copy()
    all_classes.extend(negative_classes)

    if data_preparation and print_info:
        print("Data preparation:", data_preparation)

    multivariate_dataset = dataset_name in ['reuters', 'har', 'waveform', ]

    # get dataset
    x_data, y_data = get_dataset(dataset_name)

    # class filter
    (x_data, y_data) = filter_ds(x_data, y_data, all_classes)

    # merge more classes in a single negative class (implements also subsampling)
    if len(negative_classes) > 0:
        index_neg = 0
        index_neg_to_skip = []
        dest_neg_y = np.max(all_classes) + 1  # we assure to make the negative class as the last class

        for i in range(len(y_data)):
            if y_data[i] in negative_classes:
                if index_neg % len(negative_classes) == 0:
                    y_data[i] = dest_neg_y
                else:
                    # subsampling (skip sample)
                    index_neg_to_skip.append(i)
                index_neg += 1
        x_data = np.array([x for i, x in enumerate(x_data) if i not in index_neg_to_skip])
        y_data = np.array([y for i, y in enumerate(y_data) if i not in index_neg_to_skip])

    # enumerate classes from zero to (n - 1)
    unique_y = np.sort(np.unique(y_data))
    new_y_label = 0
    for current_y_label in unique_y:
        if new_y_label != current_y_label:
            y_data[y_data == current_y_label] = new_y_label  # label replacement
        new_y_label += 1

    # determine positive and negative index class
    positive_classes = list(range(len(positive_classes)))
    negative_classes = [len(positive_classes)] if len(negative_classes) > 0 else []
    all_classes = positive_classes.copy()
    all_classes.extend(negative_classes)

    # positive and negative instances
    (x_positive, y_positive) = filter_ds(x_data, y_data, positive_classes)
    (x_negative, y_negative) = filter_ds(x_data, y_data, negative_classes)

    # 50% of negative samples in test set, 50% in train set
    index_negative_test = np.random.choice(range(len(x_negative)), int(len(x_negative) / 2), False)
    index_negative_train_unlabeled = [i for i, _ in enumerate(x_negative) if i not in index_negative_test]

    # 20% of positive samples in test set, 20% in validation, 30% labeled for training and 30% unlabeled
    index_x_positives = range(len(x_positive))

    index_positive_test = np.random.choice(index_x_positives, int(len(x_positive) / 5), False)
    index_x_positives = [x for x in index_x_positives if x not in index_positive_test] #remove indexes

    index_positive_validation = np.random.choice(index_x_positives, int(len(x_positive) / 5), False)
    index_x_positives = [x for x in index_x_positives if x not in index_positive_validation] #remove indexes

    index_positive_train_labeled = np.random.choice(index_x_positives, int(len(x_positive) / (10 / 3)), False)
    index_positive_train_unlabeled = [x for x in index_x_positives if x not in index_positive_train_labeled] # remove indexes

    # making test, validation and training sets
    x_test = np.append([x for i, x in enumerate(x_negative) if i in index_negative_test],
                       [x for i, x in enumerate(x_positive) if i in index_positive_test], axis=0)
    y_test = np.append([x for i, x in enumerate(y_negative) if i in index_negative_test],
                       [x for i, x in enumerate(y_positive) if i in index_positive_test], axis=0)

    x_val = np.array([x for i, x in enumerate(x_positive) if i in index_positive_validation])
    y_val = np.array([x for i, x in enumerate(y_positive) if i in index_positive_validation])

    x_train_labeled = np.array([x for i, x in enumerate(x_positive) if i in index_positive_train_labeled])
    y_train_labeled = np.array([x for i, x in enumerate(y_positive) if i in index_positive_train_labeled])

    x_train_unlabeled = np.append([x for i, x in enumerate(x_negative) if i in index_negative_train_unlabeled],
                                  [x for i, x in enumerate(x_positive) if i in index_positive_train_unlabeled], axis=0)
    y_train_unlabeled = np.append([x for i, x in enumerate(y_negative) if i in index_negative_train_unlabeled],
                                  [x for i, x in enumerate(y_positive) if i in index_positive_train_unlabeled], axis=0)

    # reshape data if needed
    if flatten_data or multivariate_dataset:
        x_train_labeled = x_train_labeled.reshape((len(x_train_labeled), int(np.prod(x_train_labeled.shape[1:]))))
        x_train_unlabeled = x_train_unlabeled.reshape((len(x_train_unlabeled), int(np.prod(x_train_unlabeled.shape[1:]))))
        x_test = x_test.reshape((len(x_test), int(np.prod(x_test.shape[1:]))))
        x_val = x_val.reshape((len(x_val), int(np.prod(x_val.shape[1:]))))
    elif len(x_train_labeled.shape) < 4:
        # convolutional input shape...
        x_train_labeled = x_train_labeled.reshape((len(x_train_labeled), x_train_labeled.shape[1], x_train_labeled.shape[2], 1))
        x_train_unlabeled = x_train_unlabeled.reshape((len(x_train_unlabeled), x_train_labeled.shape[1], x_train_labeled.shape[2], 1))
        x_test = x_test.reshape((len(x_test), x_train_labeled.shape[1], x_train_labeled.shape[2], 1))
        x_val = x_val.reshape((len(x_val), x_train_labeled.shape[1], x_train_labeled.shape[2], 1))

    x_train = np.concatenate((x_train_labeled, x_train_unlabeled), axis=0)

    # preprocessing data
    if data_preparation == "z_norm":
        # z-normalization
        if multivariate_dataset:
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0)

            std = np.array([x if x != 0 else 1 for x in std])  # avoid dividing by zero
        else:
            mean = np.mean(x_train)
            std = np.std(x_train)

        if print_info:
            print("Mean:", mean)
            print("Std:", std)

        x_train_labeled = (x_train_labeled - mean) / std
        x_train_unlabeled = (x_train_unlabeled - mean) / std
        x_test = (x_test - mean) / std
        x_val = (x_val - mean) / std
    elif data_preparation == "01":
        # 01 normalization
        if multivariate_dataset:
            max_ = np.max(x_train, axis=0)
            min_ = np.min(x_train, axis=0)
        else:
            max_ = np.max(x_train)
            min_ = np.min(x_train)

        if print_info:
            print("min_:", min_)
            print("max_:", max_)

        x_train_labeled = (x_train_labeled - min_) / (max_ - min_)
        x_train_unlabeled = (x_train_unlabeled - min_) / (max_ - min_)
        x_test = (x_test - min_) / (max_ - min_)
        x_val = (x_val - min_) / (max_ - min_)
    del x_train

    # cast data type
    dtype = 'float32'
    x_train_labeled = x_train_labeled.astype(dtype)
    x_train_unlabeled = x_train_unlabeled.astype(dtype)
    x_test = x_test.astype(dtype)
    x_val = x_val.astype(dtype)

    type_y = "int8"
    y_train_labeled = y_train_labeled.astype(type_y)
    y_train_unlabeled = y_train_unlabeled.astype(type_y)
    y_test = y_test.astype(type_y)
    y_val = y_val.astype(type_y)

    if print_info:
        print("Shape data:" + str(y_train_labeled[0].shape))

    # define sets based on the % of instances to take (perc_size)
    x_train_labeled = x_train_labeled[:int(len(x_train_labeled) * perc_size)]
    y_train_labeled = y_train_labeled[:int(len(y_train_labeled) * perc_size)]

    x_train_unlabeled = x_train_unlabeled[:int(len(x_train_unlabeled) * perc_size)]
    y_train_unlabeled = y_train_unlabeled[:int(len(y_train_unlabeled) * perc_size)]

    x_test = x_test[:int(len(x_test) * perc_size)]
    y_test = y_test[:int(len(y_test) * perc_size)]

    x_val = x_val[:int(len(x_val) * perc_size)]
    y_val = y_val[:int(len(y_val) * perc_size)]

    if print_info:
        print("Labeled: \t" + str(len(x_train_labeled)))
        print("Unlabeled: \t" + str(len(x_train_unlabeled)))
        print("Val set: \t" + str(len(x_val)))
        print("Test set: \t" + str(len(x_test)))

    # shuffling training sets
    shuffler1 = np.random.permutation(len(x_train_labeled))
    x_train_labeled = x_train_labeled[shuffler1]
    y_train_labeled = y_train_labeled[shuffler1]

    shuffler1 = np.random.permutation(len(x_train_unlabeled))
    x_train_unlabeled = x_train_unlabeled[shuffler1]
    y_train_unlabeled = y_train_unlabeled[shuffler1]

    return x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled, x_test, y_test, x_val, y_val


# filter dataset return samples belonging to a given set of classes
def filter_ds(x_ds, y_ds, classes):

    mask = [True if y in classes else False for y in y_ds]

    return (np.array([x_ds[i] for i, v in enumerate(mask) if v]),
        np.array([y_ds[i] for i, v in enumerate(mask) if v]))


# get number of classes for a given dataset
def get_n_classes(dataset_name):
    _, y = get_dataset(dataset_name)
    return len(np.unique(y))


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]

    data = [[float(d) for d in line.split()] for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [[float(d) for d in line.split()] for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    # images 16x16
    data_train = data_train.reshape((data_train.shape[0], 16, 16))
    data_test = data_test.reshape((data_test.shape[0], 16, 16))

    x_data = np.concatenate((data_train, data_test), axis=0)
    labels_data = np.concatenate((labels_train, labels_test), axis=0)

    return x_data, labels_data


def make_reuters_data(data_dir):
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        for did in [c for c in did_to_cat.keys()]:
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000]
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], int(x.size / x.shape[0])))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})


def load_reuters(data_path='./data/reuters'):

    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print('reutersidf saved to ' + data_path)

    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'), allow_pickle=True).item()

    x_data = data['data']
    y_data = data['label']
    x_data = x_data.reshape((x_data.shape[0], int(x_data.size / x_data.shape[0]))).astype('float64')
    y_data = y_data.reshape((y_data.size,))

    return x_data, y_data


def load_semeion():

    x_data = []
    y_data = []
    with open(os.path.join('data', 'semeion.data')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')

            x_data.append([float(d) for d in line[:256]])

            y_label = 0
            for l in line[256:]:
                if l == "1":
                    break
                else:
                    y_label += 1

            y_data.append(y_label)

    x_data = np.array(x_data)
    x_data = x_data.reshape((x_data.shape[0], int(np.sqrt(x_data.shape[1])), int(np.sqrt(x_data.shape[1]))))

    return x_data, y_data


def load_optdigits():

    def get(name):
        x_data = []
        y_data = []
        with open(os.path.join('data/optdigits', name)) as fin:
            for line in fin.readlines():
                line = line.strip().split(',')

                x_data.append([float(d.strip()) for d in line[:64]])

                y_label = int(line[64])
                y_data.append(y_label)
        x_data = np.array(x_data)
        x_data = x_data / 16.

        return x_data, y_data

    x_train, y_train = get('optdigits.tra')
    x_test, y_test = get('optdigits.tes')

    x_data = np.concatenate((x_train, x_test), axis=0)
    labels_data = np.concatenate((y_train, y_test), axis=0)

    return x_data, labels_data


def load_har():

    def get(name_x, name_y):
        x_data = []
        y_data = []
        with open(os.path.join('data/har', name_x)) as fin:
            for line in fin.readlines():
                line = np.array(line.strip().split())
                x_data.append([float(d.strip()) for d in line[:561]])

        with open(os.path.join('data/har', name_y)) as fin:
            for line in fin.readlines():
                y_data.append(int(line.strip()) - 1) # tra 0 e 5

        x_data = np.array(x_data)

        return x_data, y_data

    x_train, y_train = get('train/X_train.txt', 'train/y_train.txt')
    x_test, y_test = get('test/X_test.txt', 'test/y_test.txt')

    x_data = np.concatenate((x_train, x_test), axis=0)
    labels_data = np.concatenate((y_train, y_test), axis=0)

    return x_data, labels_data


def load_waveform():
    x_data = []
    y_data = []
    with open(os.path.join('data', 'waveform-5000_csv.csv')) as fin:
        for line in fin.readlines()[1:]:
            line = line.strip().split(',')

            x_data.append([float(d) for d in line[:40]])

            y_label = int(line[40])
            y_data.append(y_label)

    x_data = np.array(x_data)

    return x_data, y_data


def load_pendigits():

    def get(name):
        x_data = []
        y_data = []
        with open(os.path.join('data/pendigits', name)) as fin:
            for line in fin.readlines():
                line = line.strip().split(',')

                x_data.append([float(d.strip()) for d in line[:16]])

                y_label = int(line[16])
                y_data.append(y_label)
        x_data = np.array(x_data)
        x_data = x_data / 100.  # normalization

        return x_data, y_data

    x_train, y_train = get('pendigits.tra')
    x_test, y_test = get('pendigits.tes')

    x_data = np.concatenate((x_train, x_test), axis=0)
    labels_data = np.concatenate((y_train, y_test), axis=0)

    return x_data, labels_data


def load_sonar():
    x_data = []
    y_data = []

    with open('data/sonar.all-data') as fin:
        for line in fin.readlines():
            line = line.strip().split(',')

            x_data.append([float(d.strip()) for d in line[:60]])

            y_label = 0 if line[60] == 'R' else 1
            y_data.append(y_label)

    return x_data, y_data


def load_landsat():

    def get(name):
        x_data = []
        y_data = []
        with open(os.path.join('data/landsat', name)) as fin:
            for line in fin.readlines():
                line = line.strip().split(' ')

                x_data.append([float(d.strip()) for d in line[:36]])

                y_label = int(line[36])

                y_data.append(5 if y_label == 7 else y_label - 1)
        x_data = np.array(x_data)

        return x_data, y_data

    x_train, y_train = get('sat.trn')
    x_test, y_test = get('sat.tst')

    x_data = np.concatenate((x_train, x_test), axis=0)
    labels_data = np.concatenate((y_train, y_test), axis=0)

    return x_data, labels_data
