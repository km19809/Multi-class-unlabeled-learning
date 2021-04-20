import numpy as np
import tensorflow.compat.v1 as tf
import os
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


def get_dataset_info(dataset_name):
    x, y = get_dataset(dataset_name)
    print("{}: {} samples, {} classes".format(dataset_name, len(x), len(np.unique(y))))


dataset_repo = dict()


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

    # shuffle data with the same permutation
    if dataset_name not in dataset_repo:
        dataset_repo[dataset_name] = np.random.permutation(len(x_data))

    shuffler = dataset_repo[dataset_name]

    x_data = np.array(x_data)[shuffler]
    y_data = np.array(y_data)[shuffler]

    return x_data, y_data


# restituisce il dataset Mnist suddiviso in esempi etichettati e non, più il test set
def get_data(positive_classes, negative_class, perc_labeled, k_fold, flatten_data=False,
             perc_size=1, dataset_name="mnist", perc_test_set=0.2, perc_val_set=0.2,
             data_preparation=None, print_some=False):

    all_class = positive_classes.copy()
    all_class.extend(negative_class)

    if data_preparation and print_some:
        print("Data preparation:", data_preparation)

    multivariate_dataset = dataset_name in ['reuters', 'har', 'waveform', ]

    # get dataset
    x_data, y_data = get_dataset(dataset_name)

    # filtro per classe
    (x_data, y_data) = filter_ds(x_data, y_data, all_class)

    # li esempi negativi li si fanno confluire tutti in un'unica classe e si effettua un sottocampionamento
    if len(negative_class) > 1:
        index_neg = 0
        index_neg_to_skip = []
        dest_neg_y = np.min(negative_class)

        for i in range(len(y_data)):
            if y_data[i] in negative_class:
                if index_neg % len(negative_class) == 0:
                    y_data[i] = dest_neg_y
                else:
                    # si scarta l'esempio negativo (sottocampionamento)
                    index_neg_to_skip.append(i)
                index_neg += 1
        x_data = [x for i, x in enumerate(x_data) if i not in index_neg_to_skip]
        y_data = [y for i, y in enumerate(y_data) if i not in index_neg_to_skip]

    # ottenimento train e test set in base a K
    tot_test = len(x_data) * perc_test_set
    tot_val = len(x_data) * perc_val_set

    test_begin_index = tot_test * k_fold
    test_end_index = tot_test * (k_fold + 1)

    x_test = np.array([x for i, x in enumerate(x_data) if test_begin_index <= i < test_end_index])
    y_test = np.array([x for i, x in enumerate(y_data) if test_begin_index <= i < test_end_index])

    val_begin_index = tot_val * (k_fold + 1)
    if val_begin_index >= len(x_data):
        val_begin_index = 0
        val_end_index = tot_val
    else:
        val_end_index = tot_val * (k_fold + 2)

    x_val = np.array([x for i, x in enumerate(x_data) if val_begin_index <= i < val_end_index])
    y_val = np.array([x for i, x in enumerate(y_data) if val_begin_index <= i < val_end_index])

    x_train = np.array([x for i, x in enumerate(x_data) if not (test_begin_index <= i < test_end_index) and not (val_begin_index <= i < val_end_index)])
    y_train = np.array([x for i, x in enumerate(y_data) if not (test_begin_index <= i < test_end_index) and not (val_begin_index <= i < val_end_index)])

    # modifiche per corretta elaborazione dei dati
    if flatten_data or multivariate_dataset:
        x_train = x_train.reshape((len(x_train), int(np.prod(x_train.shape[1:]))))
        x_test = x_test.reshape((len(x_test), int(np.prod(x_test.shape[1:]))))
        x_val = x_val.reshape((len(x_val), int(np.prod(x_val.shape[1:]))))
    else:
        # per la convoluzionale (ogni input deve avere sempre 3 dimensioni)
        if len(x_train.shape) < 4:
            x_train = x_train.reshape((len(x_train), x_train.shape[1], x_train.shape[2], 1))
            x_test = x_test.reshape((len(x_test), x_train.shape[1], x_train.shape[2], 1))
            x_val = x_val.reshape((len(x_val), x_train.shape[1], x_train.shape[2], 1))

    # preprocessing data
    if data_preparation == "z_norm":
        if multivariate_dataset:
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0)
        else:
            mean = np.mean(x_train)
            std = np.std(x_train)

        if print_some:
            print("Mean:", mean)
            print("Std:", std)

        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_val = (x_val - mean) / std
    elif data_preparation == "01":
        if multivariate_dataset:
            max_ = np.max(x_train, axis=0)
            min_ = np.min(x_train, axis=0)
        else:
            max_ = np.max(x_train)
            min_ = np.min(x_train)

        if print_some:
            print("min_:", min_)
            print("max_:", max_)

        x_train = (x_train - min_) / (max_ - min_)
        x_test = (x_test - min_) / (max_ - min_)
        x_val = (x_val - min_) / (max_ - min_)

    dtype = 'float32'
    x_train = x_train.astype(dtype)
    x_test = x_test.astype(dtype)
    x_val = x_val.astype(dtype)

    type_y = "int8"
    y_train = y_train.astype(type_y)
    y_test = y_test.astype(type_y)
    y_val = y_val.astype(type_y)

    # esempi positivi
    (x_train_positive, y_train_positive) = filter_ds(x_train, y_train, positive_classes)

    # esempi negativi
    (x_train_negative, y_train_negative) = filter_ds(x_train, y_train, negative_class)

    tot_labeled = int(len(x_train_positive) * perc_labeled)

    # dataset che contiene gli esempi etichettati
    x_train_labeled = np.array([x for i, x in enumerate(x_train_positive) if i < tot_labeled])
    y_train_labeled = np.array([y for i, y in enumerate(y_train_positive) if i < tot_labeled])

    shuffler1 = np.random.permutation(len(x_train_labeled))
    x_train_labeled = x_train_labeled[shuffler1]
    y_train_labeled = y_train_labeled[shuffler1]

    # esempi non etichettati (comprende gli esempi positivi e quelli negativi)
    x_train_unlabeled = np.array([x for i, x in enumerate(x_train_positive) if i >= tot_labeled])
    y_train_unlabeled = np.array([y for i, y in enumerate(y_train_positive) if i >= tot_labeled])

    x_train_unlabeled = np.append(x_train_unlabeled, x_train_negative, axis=0)
    y_train_unlabeled = np.append(y_train_unlabeled, y_train_negative, axis=0)

    shuffler1 = np.random.permutation(len(x_train_unlabeled))
    x_train_unlabeled = x_train_unlabeled[shuffler1]
    y_train_unlabeled = y_train_unlabeled[shuffler1]

    if print_some:
        print("Shape data:" + str(x_train[0].shape))
        #print("Shape y data:" + str(y_train[0].shape))

    x_train_labeled = x_train_labeled[:int(len(x_train_labeled) * perc_size)]
    x_train_unlabeled = x_train_unlabeled[:int(len(x_train_unlabeled) * perc_size)]
    x_train_positive = x_train_positive[:int(len(x_train_positive) * perc_size)]
    x_train_negative = x_train_negative[:int(len(x_train_negative) * perc_size)]
    y_train_unlabeled = y_train_unlabeled[:int(len(y_train_unlabeled) * perc_size)]
    y_train_labeled = y_train_labeled[:int(len(y_train_labeled) * perc_size)]

    x_train = x_train[:int(len(x_train) * perc_size)]
    y_train = y_train[:int(len(y_train) * perc_size)]
    x_test = x_test[:int(len(x_test) * perc_size)]
    y_test = y_test[:int(len(y_test) * perc_size)]
    x_val = x_val[:int(len(x_val) * perc_size)]
    y_val = y_val[:int(len(y_val) * perc_size)]

    if print_some:
        print("\nTotal train: \t\t" + str(len(x_train)))
        print("Labeled: \t" + str(len(x_train_labeled)))
        print("Unlabeled: \t" + str(len(x_train_unlabeled)))
        print("Positive: \t" + str(len(x_train_positive)))
        print("Negative: \t" + str(len(x_train_negative)))
        print("Val set: \t" + str(len(x_val)))
        print("Test set: \t" + str(len(x_test)))

        print("Train")
        for c in all_class:
            print("Class:", c, "->",  len(filter_ds(x_train, y_train, [c])[0]))
        print("Test")
        for c in all_class:
            print("Class:", c, "->",  len(filter_ds(x_test, y_test, [c])[0]))

    return x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled, x_test, y_test, x_val, y_val


def filter_ds(x_ds, y_ds, classes):

    mask = [True if y in classes else False for y in y_ds]

    return (np.array([x_ds[i] for i, v in enumerate(mask) if v]),
        np.array([y_ds[i] for i, v in enumerate(mask) if v]))


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

    # immagine 16x16
    data_train = data_train.reshape((data_train.shape[0], 16, 16))
    data_test = data_test.reshape((data_test.shape[0], 16, 16))

    # tutto per il training
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
    # has been shuffled
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
        x_data = x_data / 100. #normalizzation

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

                # si fa in modo di avere le classi comprese tra 0 e N - 1
                y_data.append(5 if y_label == 7 else y_label - 1)
        x_data = np.array(x_data)

        return x_data, y_data

    x_train, y_train = get('sat.trn')
    x_test, y_test = get('sat.tst')

    x_data = np.concatenate((x_train, x_test), axis=0)
    labels_data = np.concatenate((y_train, y_test), axis=0)

    return x_data, labels_data


def get_n_classes(dataset_name):
    _, y = get_dataset(dataset_name)
    return len(np.unique(y))
