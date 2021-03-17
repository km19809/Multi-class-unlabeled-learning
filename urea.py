import numpy as np
import matplotlib.pyplot as plt
import custom_layers, get_data
import keras
from keras import Model, Input
from keras.layers import Dense
from keras import layers
from keras.optimizers import SGD, Adam
from sklearn.manifold import TSNE
import tensorflow as tf


positive_classes = [0,1,2,3,4,5,6,7,8]
negative_classes = [9]

k = 10

n_unlabeled = 1


positive_class_factors = tf.constant([0.2,0.1,0.1,0.3,0.1,0.1,0.1,0.1,0.9,], dtype='float32')


def hinge(y_val):
    return tf.maximum(0., 1. - y_val)

def labeled_loss(y_true, y_pred):

    #y_pred[:k - 1] = y_pred[:k - 1] * y_true #si azzerano i valori diversi dalla classe di appartenenza
    #y_pred[k] = y_pred[k] * -1

    #y_pred = hinge(y_pred)
    # y_pred = tf.reduce_sum(y_pred, 1)

    y_pred_pos = y_pred[:, :k - 1] * y_true
    y_pred_pos = tf.reduce_sum(y_pred_pos, 1)
    y_pred_pos = hinge(y_pred_pos)

    y_pred_neg = hinge(y_pred[:, k - 1] * -1)

    y_pred = y_pred_pos + y_pred_neg

    factors = tf.linalg.tensordot(y_true, positive_class_factors, axes=1)
    y_pred = y_pred * factors

    y_pred = tf.reduce_sum(y_pred)

    return y_pred * k / (k - 1)


unl_1 = tf.constant([-1,-1,-1,-1,-1,-1,-1,-1,-1,1], dtype='float32')
unl_2 = tf.constant([k-1,k-1,k-1,k-1,k-1,k-1,k-1,k-1,k-1,1], dtype='float32')


def unlabeled_loss(y_true, y_pred):

    y_pred = tf.math.multiply(y_pred, unl_1)
    #y_pred_1 = y_pred[:,:k-1] * -1

    y_pred = hinge(y_pred)

    #y_pred = tf.math.multiply(y_pred, unl_2)
    y_pred  =y_pred / unl_2
    #y_pred[:k - 1] = y_pred[:k - 1] / (k - 1)

    y_pred = tf.reduce_sum(y_pred)

    return y_pred / n_unlabeled


def create_model(input_dim):

    weight_decay = 0.001
    dims = [ 250, 250, 1000, 10]

    input = Input(shape=(input_dim, ))
    l = input

    for d in dims:
        act = 'relu' if d != dims[-1] else None
        l = Dense(d, activation=act,
                  kernel_regularizer=keras.regularizers.l2(weight_decay), bias_regularizer=keras.regularizers.l2(weight_decay),)(l)

    model_unlabeled = Model(input, l)
    model_labeled = Model(input, l)

    # inserire pesi
    model_unlabeled.compile(SGD(learning_rate=0.05), unlabeled_loss, )
    model_labeled.compile(SGD(learning_rate=0.05), labeled_loss, )

    return model_unlabeled, model_labeled


def get_dataset():
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val =\
    get_data.get_data(positive_classes, negative_classes, 0.5, True, dataset_name="pendigits", data_preparation=False)

    return ds_labeled, y_labeled, ds_unlabeled, y_unlabeled




def main():

    #dataset
    ds_labeled, y_labeled, ds_unlabeled, y_unlabeled = get_dataset()

    ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
    y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

    # priors and so on
    global n_unlabeled, positive_class_factors
    n_unlabeled = len(ds_unlabeled)

    priors = []
    n_elements = []
    for p_c in positive_classes:
        els_class, _ = get_data.filter_ds(ds_all, y_all, [p_c])
        els_class_labeled, _ = get_data.filter_ds(ds_labeled, y_labeled, [p_c])

        prior = len(els_class) / len(ds_all)
        n_labeled = len(els_class_labeled)

        priors.append(prior)
        n_elements.append(n_labeled)

    tmp = np.array(priors) / np.array(n_elements)
    positive_class_factors = tf.constant(tmp, dtype='float32')

    # models
    input_dim = ds_unlabeled[0].shape[0]
    model_unlabeled, model_labeled = create_model(input_dim)

    # train
    batch_size = 256
    if batch_size > len(ds_labeled):
        batch_size = len(ds_labeled)

    epochs = 200

    y_labeled_categorical = keras.utils.to_categorical(y_labeled)

    for e in range(epochs):

        print("Epoch:", e + 1, "/", epochs)

        # shuffle
        shuffler1 = np.random.permutation(len(ds_labeled))
        ds_labeled = ds_labeled[shuffler1]
        y_labeled = y_labeled[shuffler1]
        y_labeled_categorical = y_labeled_categorical[shuffler1]

        shuffler1 = np.random.permutation(len(ds_unlabeled))
        ds_unlabeled = ds_unlabeled[shuffler1]
        y_unlabeled = y_unlabeled[shuffler1]

        index_unlabeled = 0
        index_labeled = 0
        finish_unlabeled = False
        finish_labeled = False

        while not finish_unlabeled or not finish_labeled:
            # batch unlabeled
            if not finish_unlabeled:
                if (index_unlabeled + 1) * batch_size >= ds_unlabeled.shape[0]:
                    loss = model_unlabeled.train_on_batch(x=ds_unlabeled[index_unlabeled * batch_size::],
                                                          y=ds_unlabeled[index_unlabeled * batch_size:(index_unlabeled + 1) * batch_size])

                    print("loss U:", np.round(loss, 5))
                    finish_unlabeled = True
                else:
                    loss = model_unlabeled.train_on_batch(x=ds_unlabeled[index_unlabeled * batch_size:(index_unlabeled + 1) * batch_size],
                        y=ds_unlabeled[index_unlabeled * batch_size:(index_unlabeled + 1) * batch_size])
                    index_unlabeled += 1

            # batch LABELED
            if not finish_labeled:
                if (index_labeled + 1) * batch_size >= ds_labeled.shape[0]:
                    loss = model_labeled.train_on_batch(x=ds_labeled[index_labeled * batch_size::],
                                                          y=y_labeled_categorical[index_labeled * batch_size::])

                    print("loss L:", np.round(loss, 5))
                    finish_labeled = True
                else:
                    loss = model_labeled.train_on_batch(
                        x=ds_labeled[index_labeled * batch_size:(index_labeled + 1) * batch_size],
                        y=y_labeled_categorical[index_labeled * batch_size:(index_labeled + 1) * batch_size])
                    index_labeled += 1


    model_labeled.save_weights("parameters/UREA.h5")
    # finish train

    # accuracy
    y_pred = model_labeled.predict(ds_all)
    y_pred = np.argmax(y_pred, axis=1)

    acc = sum([1 for i, _ in enumerate(y_pred) if y_all[i] == y_pred[i]]) * 100. / len(ds_all)

    print("ACCURACY is:", acc, "%")



if __name__ == '__main__':
    main()


