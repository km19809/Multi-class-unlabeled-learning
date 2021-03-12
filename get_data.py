import numpy as np
import tensorflow.compat.v1 as tf


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
    elif dataset_name == "ups":
        (x_train, y_train), (x_test, y_test) = load_usps()
    elif dataset_name == "reuters":
        (x_train, y_train), (x_test, y_test) = load_reuters()
    else: #mnist
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # filtro per classe
    (x_train, y_train) = filter_ds(x_train, y_train, all_class)
    (x_test, y_test) = filter_ds(x_test, y_test, all_class)

    # modifiche per corretta elaborazione dei dati

    if flatten_data:
        x_train = x_train.reshape((len(x_train), int(np.prod(x_train.shape[1:]))))
        x_test = x_test.reshape((len(x_test), int(np.prod(x_test.shape[1:]))))

        # preprocessing z score
        if data_preparation:
            mean, std = get_mean_std(x_train, axis=None)

            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std
    else:
        # per la convoluzionale (ogni input deve avere sempre 3 dimensioni)
        if len(x_train.shape) < 4:
            x_train = x_train.reshape((len(x_train), x_train.shape[1], x_train.shape[2], 1))
            x_test = x_test.reshape((len(x_test), x_train.shape[1], x_train.shape[2], 1))

        # preprocessing z score
        if data_preparation:
            mean, std = get_mean_std(x_train)

            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std

    dtype = 'float32'
    x_train = x_train.astype(dtype)
    x_test = x_test.astype(dtype)

    type_y = "int8"
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
    for c in all_class:
        print("Class:", c, "->",  len(filter_ds(x_train, y_train, [c])[0]))

    return x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled, x_test, y_test


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
    data_train = np.concatenate((data_train, data_test), axis=0)
    labels_train = np.concatenate((labels_train, labels_test), axis=0)

    data_test = []
    labels_test = []

    return (data_train, labels_train), (data_test, labels_test)


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
    import os
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

    shuffler1 = np.random.permutation(len(x_data))
    x_data = x_data[shuffler1]
    y_data = y_data[shuffler1]

    perc_for_validation = 0.0001 #tutti in training

    tot_train = int(len(x_data) * perc_for_validation)

    # suddivisione in test e train
    x_test = np.array([x for i, x in enumerate(x_data) if i < tot_train])
    y_test = np.array([y for i, y in enumerate(y_data) if i < tot_train])

    x_train = np.array([x for i, x in enumerate(x_data) if i >= tot_train])
    y_train = np.array([y for i, y in enumerate(y_data) if i >= tot_train])

    return (x_train, y_train), (x_test, y_test)





