import base_classifier as bc
import numpy as np
from keras.layers import Input, Dense
from keras import Model
import keras
from keras.optimizers import SGD
import tensorflow as tf
import get_data


class LinearSVM(bc.BaseClassifier):

    def get_model(self):
        from sklearn.svm import LinearSVC

        model = LinearSVC()

        return model

    def single_run(self, current_run):

        # dataset
        ds_labeled, y_labeled, _, _, x_val, y_val = self.get_dataset()

        # model
        model = self.get_model()

        # train model
        model.fit(ds_labeled, y_labeled)

        # predict labels (test and train)
        y_pred_test = model.predict(x_val)
        y_pred_train = model.predict(ds_labeled)

        # get accuracy
        test_mes = self.get_accuracy(y_pred_test, y_val)
        train_mes = self.get_accuracy(y_pred_train, y_labeled)

        return train_mes, test_mes


class AREA(bc.BaseClassifier):

    def hinge(self, y_val):
        return tf.maximum(0., 1. - y_val)

    def labeled_loss(self, y_true, y_pred):

        k = len(self.positive_classes) + 1

        y_pred_pos = y_pred[:, :k - 1] * y_true
        y_pred_pos = tf.reduce_sum(y_pred_pos, 1)
        y_pred_pos = self.hinge(y_pred_pos)

        y_pred_neg = self.hinge(y_pred[:, k - 1] * -1)

        y_pred = y_pred_pos + y_pred_neg

        factors = tf.linalg.tensordot(y_true, self.positive_class_factors, axes=1)
        y_pred = y_pred * factors

        y_pred = tf.reduce_sum(y_pred)

        return y_pred * k / (k - 1)

    def unlabeled_loss(self, y_true, y_pred):
        # ottenimento coefficienti
        k = len(self.positive_classes) + 1
        coeff_1 = []
        coeff_2 = []
        for i in range(len(self.positive_classes)):
            coeff_1.append(-1.)
            coeff_2.append(k - 1)
        coeff_1.append(1.)
        coeff_2.append(1.)

        unl_1 = tf.constant(coeff_1, dtype='float32')
        unl_2 = tf.constant(coeff_2, dtype='float32')

        # calcolo
        y_pred = tf.math.multiply(y_pred, unl_1)

        y_pred = self.hinge(y_pred)

        y_pred = y_pred / unl_2

        y_pred = tf.reduce_sum(y_pred)

        return y_pred / self.n_unlabeled

    def get_model(self):
        weight_decay = 0.001
        dims = [1000, 500, 250, 10]

        input = Input(shape=(self.input_dim,))
        l = input

        for d in dims:
            act = 'relu' if d != dims[-1] else None
            l = Dense(d, activation=act,
                      kernel_regularizer=keras.regularizers.l2(weight_decay),
                      bias_regularizer=keras.regularizers.l2(weight_decay), )(l)

        model_unlabeled = Model(input, l)
        model_labeled = Model(input, l)

        # inserire pesi
        model_unlabeled.compile(SGD(), self.unlabeled_loss, )
        model_labeled.compile(SGD(), self.labeled_loss, )

        return model_unlabeled, model_labeled

    def single_run(self, current_run):

        # dataset
        ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = self.get_dataset()

        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        # model
        self.input_dim = ds_unlabeled[0].shape[0]
        self.n_unlabeled = len(ds_unlabeled)

        # ottenimento probabilità a priori (todo ora vengono lette dal dataset)
        priors = []
        n_elements = []
        for p_c in self.positive_classes:
            els_class, _ = get_data.filter_ds(ds_all, y_all, [p_c])
            els_class_labeled, _ = get_data.filter_ds(ds_labeled, y_labeled, [p_c])

            prior = len(els_class) / len(ds_all)
            n_labeled = len(els_class_labeled)

            priors.append(prior)
            n_elements.append(n_labeled)

        tmp = np.array(priors) / np.array(n_elements)
        self.positive_class_factors = tf.constant(tmp, dtype='float32')
        del tmp

        model_unlabeled, model_labeled = self.get_model()


        # train model
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
                                                              y=ds_unlabeled[index_unlabeled * batch_size:(
                                                                                                                      index_unlabeled + 1) * batch_size])

                        print("loss U:", np.round(loss, 5))
                        finish_unlabeled = True
                    else:
                        loss = model_unlabeled.train_on_batch(
                            x=ds_unlabeled[index_unlabeled * batch_size:(index_unlabeled + 1) * batch_size],
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

        # predict labels (test and train)
        y_pred_test = np.argmax(model_labeled.predict(x_val), axis=1)
        y_pred_train = np.argmax(model_labeled.predict(ds_all), axis=1)

        # get accuracy
        test_mes = self.get_accuracy(y_pred_test, y_val)
        train_mes = self.get_accuracy(y_pred_train, y_all)

        return train_mes, test_mes


if __name__ == '__main__':

    n_runs = 3
    perc_ds = 1
    perc_labeled = 0.5
    data_preparation = False

    for dataset_name in ["optdigits", "pendigits", "usps"]:
        for name in ["area"]:
            # get model
            if name == "linearSVM":
                model = LinearSVM(n_runs, dataset_name, perc_ds, 1, data_preparation, name)
            elif name == "area":
                model = AREA(n_runs, dataset_name, perc_ds, perc_labeled, data_preparation, name)
            else:
                model = None

            # get measures
            train_mes, test_mes = model.train()

            mean_mes = np.mean(test_mes)
            format = "{:5.3f}"

            print(name, "-> TEST:", format.format(np.mean(test_mes)), "+-", format.format(np.std(test_mes)))
            print("TRAIN:", format.format(np.mean(train_mes)), "+-", format.format(np.std(train_mes)))