import base_classifier as bc
import numpy as np
from keras.layers import Input, Dense
from keras import Model
import keras
from keras.optimizers import SGD, Adam
import tensorflow as tf
import get_data
from sklearn.model_selection import KFold
from keras import backend as K


class LinearSVM(bc.BaseClassifier):

    def get_model(self):
        from sklearn.svm import LinearSVC

        model = LinearSVC(dual=False)

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

    def the_loss(self, y_true, y_pred):

        # calcolo (y true servono solo come fattori)
        y_pred = y_pred * y_true[:, 0]
        y_pred = tf.maximum(0., 1. - y_pred)
        y_pred = y_pred * y_true[:, 1]
        y_pred = tf.reduce_sum(y_pred)

        return y_pred

    def get_model(self):
        dims = [10]

        input = Input(shape=(self.input_dim,))
        l = input

        for d in dims:
            act = 'relu' if d != dims[-1] else None
            l = Dense(d, activation=act,
                      kernel_regularizer=keras.regularizers.l2(self.hyper_parameters['weight_decay']),)(l)

        model = Model(input, l)

        model.compile(Adam(self.hyper_parameters['learning_rate']), self.the_loss)

        return model

    def single_run(self, current_run):

        # dataset
        ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = self.get_dataset()

        # determinazione dei fattori da utilizzare per la loss
        K = len(self.classes)
        N_U = len(ds_unlabeled)

        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        # ottenimento probabilità a priori (todo ora vengono lette dal dataset)
        positive_class_factors = []
        for p_c in self.positive_classes:
            els_class, _ = get_data.filter_ds(ds_all, y_all, [p_c])
            els_class_labeled, _ = get_data.filter_ds(ds_labeled, y_labeled, [p_c])

            prior = len(els_class) / len(ds_all)
            n_labeled = len(els_class_labeled)

            positive_class_factors.append(prior / n_labeled)

        product_function = []
        product_loss = []

        # fattori per gli esempi non etichettati (è importante come viene definito ds_all)
        for i in range(N_U):
            product_function.append([-1 if k != len(self.classes) - 1 else 1 for k, _ in enumerate(self.classes)])  # vettore di -1 o +1 alla fine
            product_loss.append([1. / N_U if k == len(self.classes) - 1 else 1 / (N_U * (K - 1)) for k, _ in enumerate(self.classes)])  #

        # fattori per gli esempi etichettati (vedere la definizione nel paper)
        for y in y_labeled:
            product_function.append([1 if k != len(self.classes) -1 else -1 for k, _ in enumerate(self.classes)]) # vettore di +1 o -1 alla fine

            p_l = (K / (K - 1)) * positive_class_factors[y]
            product_loss.append([p_l if k == y or k == len(self.classes) -1 else 0 for k, _ in enumerate(self.classes)]) #

        factors = []
        for i in range(len(product_loss)):
            factors.append([product_function[i], product_loss[i]])
        factors = np.array(factors).reshape((len(product_loss), 2, K))

        # train model for different hyperparameters
        best_hyperparameters = {
            'weight_decay': 1e-5,
            'learning_rate': 1e-4,
        }
        best_accuracy = 0

        nums = [#1e-5,1e-4,1e-3,1e-2,1e-1,
         ]
        for w_d in nums:
            for l_r in nums:

                # get model
                self.hyper_parameters['weight_decay'] = w_d
                self.hyper_parameters['learning_rate'] = l_r
                model_unlabeled, model_labeled = self.get_model()

                # get folds
                kf = KFold(n_splits=self.num_folds)
                labeled_split = [s for s in kf.split(ds_labeled)]
                unlabeled_split = [s for s in kf.split(ds_unlabeled)]

                test_measures_fold = 0
                for k in range(self.num_folds):
                    #print("Current fold:", k)

                    # ottenimento split
                    labeled_train_index, labeled_test_index = labeled_split[k]
                    unlabeled_train_index, unlabeled_test_index = unlabeled_split[k]

                    # train model
                    labeled_x = ds_labeled[labeled_train_index]
                    labeled_y = y_labeled[labeled_train_index]
                    unlabeled_x = ds_unlabeled[unlabeled_train_index]

                    self.fit(labeled_y, labeled_x, unlabeled_x, model_labeled, model_unlabeled)

                    # get accuracy for fold
                    test_fold = ds_labeled[labeled_test_index]
                    y_test_fold = y_labeled[labeled_test_index]

                    y_pred_fold = np.argmax(model_labeled.predict(test_fold), axis=1)
                    test_fold_mes = self.get_accuracy(y_pred_fold, y_test_fold)
                    test_measures_fold += test_fold_mes

                test_measures_fold /= self.num_folds

                # determinare se questo modello è il piu accurato
                if test_measures_fold > best_accuracy:
                    best_accuracy = test_measures_fold
                    best_hyperparameters = self.hyper_parameters

        # allenamento col miglior modello trovato
        #print("Best hyper parameters:", best_hyperparameters)
        self.hyper_parameters = best_hyperparameters
        model = self.get_model()
        model.fit(ds_all, factors, batch_size=256, epochs=1000, shuffle=True, verbose=0)

        # TESTING FINALE
        # predict labels (test and train)
        y_pred_test = np.argmax(model.predict(x_val), axis=1)
        y_pred_train = np.argmax(model.predict(ds_all), axis=1)

        # get accuracy
        test_mes = self.get_accuracy(y_pred_test, y_val)
        train_mes = self.get_accuracy(y_pred_train, y_all)

        return train_mes, test_mes


class UREA(bc.BaseClassifier):

    def the_loss(self, y_true, y_pred):

        y_pred_pos = tf.maximum(0., 1. - y_pred)
        y_pred_neg = tf.maximum(0., 1. - (-y_pred))

        res = y_pred_pos * y_true[:, 0] # calcolo per hinge(z)
        res += y_pred_neg * y_true[:, 1] # calcolo per hinge(-z)

        res = tf.reduce_sum(res)

        return res

    def get_model(self):
        dims = [10]

        input = Input(shape=(self.input_dim,))
        l = input

        for d in dims:
            act = 'relu' if d != dims[-1] else None
            l = Dense(d, activation=act,
                      kernel_regularizer=keras.regularizers.l2(self.hyper_parameters['weight_decay']),)(l)

        model = Model(input, l)

        model.compile(Adam(self.hyper_parameters['learning_rate']), self.the_loss)

        return model

    def single_run(self, current_run):

        # dataset
        ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = self.get_dataset()

        # determinazione dei fattori da utilizzare per la loss
        K = len(self.classes)
        N_U = len(ds_unlabeled)

        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        # ottenimento probabilità a priori (todo ora vengono lette dal dataset)
        positive_class_factors = []
        for p_c in self.positive_classes:
            els_class, _ = get_data.filter_ds(ds_all, y_all, [p_c])
            els_class_labeled, _ = get_data.filter_ds(ds_labeled, y_labeled, [p_c])

            prior = len(els_class) / len(ds_all)
            n_labeled = len(els_class_labeled)

            positive_class_factors.append(prior / n_labeled)

        product_loss_pos = []
        product_loss_neg = []

        # fattori per gli esempi non etichettati (è importante come viene definito ds_all)
        for i in range(N_U):
            product_loss_pos.append([1. / N_U if k == len(self.classes) - 1 else 0 for k, _ in enumerate(self.classes)])  #
            product_loss_neg.append([0 if k == len(self.classes) - 1 else 1 / (N_U * (K - 1)) for k, _ in enumerate(self.classes)])  #

        # fattori per gli esempi etichettati (vedere la definizione nel paper)
        for y in y_labeled:
            p_l = positive_class_factors[y]
            product_loss_pos.append([p_l if k == y else -p_l if k == len(self.classes) -1 else 0 for k, _ in enumerate(self.classes)])

            p_l = p_l / (K - 1)
            product_loss_neg.append([-p_l if k == y else p_l if k == len(self.classes) -1 else 0 for k, _ in enumerate(self.classes)])

        factors = []
        for i in range(len(product_loss_pos)):
            factors.append([product_loss_pos[i], product_loss_neg[i]])
        factors = np.array(factors).reshape((len(product_loss_pos), 2, K))

        # train model for different hyperparameters
        best_hyperparameters = {
            'weight_decay': 1e-5,
            'learning_rate': 1e-4,
        }

        # allenamento col miglior modello trovato
        #print("Best hyper parameters:", best_hyperparameters)
        self.hyper_parameters = best_hyperparameters
        model = self.get_model()
        model.fit(ds_all, factors, batch_size=256, epochs=1000, shuffle=True, verbose=0)

        # TESTING FINALE
        # predict labels (test and train)
        y_pred_test = np.argmax(model.predict(x_val), axis=1)
        y_pred_train = np.argmax(model.predict(ds_all), axis=1)

        # get accuracy
        test_mes = self.get_accuracy(y_pred_test, y_val)
        train_mes = self.get_accuracy(y_pred_train, y_all)

        return train_mes, test_mes


class UPU(bc.BaseClassifier):

    def the_loss(self, y_true, y_pred):

        # calcolo (y true servono solo come fattori)
        labeled_loss = y_true[:, 0] * y_pred
        unlabeled_loss = y_true[:, 1] * tf.math.log(1. + tf.math.exp(y_pred))

        return tf.reduce_sum(labeled_loss + unlabeled_loss)

    def get_model(self):
        input = Input(shape=(self.input_dim,))
        l = Dense(1, activation="linear", kernel_initializer='glorot_uniform',
                  kernel_regularizer=keras.regularizers.l2(self.hyper_parameters['weight_decay']), )(input)

        model = Model(input, l)
        model.compile(Adam(self.hyper_parameters['learning_rate']), self.the_loss)

        return model

    def single_run(self, current_run):
        # dataset
        ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = self.get_dataset()

        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        centers_radial = ds_all.copy()

        # ottenimento prob. a priori per le classi positive (todo ora vengono lette dal dataset)
        positive_priors = dict()
        for p_c in self.positive_classes:
            els_class, _ = get_data.filter_ds(ds_all, y_all, [p_c])
            prior = len(els_class) / len(ds_all)
            positive_priors[p_c] = prior

        sigma = np.var(ds_all, axis=0)
        sigma = np.mean(sigma)

        self.input_dim = len(ds_all)
        self.hyper_parameters['weight_decay'] = 1e-6
        self.hyper_parameters['learning_rate'] = 0.01

        # modelli
        models = dict()

        # allenare dataset per k-1 classi
        for k in self.positive_classes:
            #print("K positive=", k)

            # esempi etichettati della k-esima classe
            single_ds_labeled, _ = get_data.filter_ds(ds_labeled, y_labeled, [k])

            # esempi non etichettati
            single_ds_unlabeled, _ = get_data.filter_ds(ds_labeled, y_labeled, [c for c in self.positive_classes if c != k])
            single_ds_unlabeled = np.concatenate((single_ds_unlabeled, ds_unlabeled), axis=0)

            single_ds_all = np.concatenate((single_ds_labeled, single_ds_unlabeled), axis=0)

            #
            class_prior = positive_priors[k]
            n_unlabeled = len(single_ds_unlabeled)
            n_labeled = len(single_ds_labeled)

            # calcolo matrice esponenziale
            rbf = self.get_rbf(centers_radial, single_ds_all, sigma)

            # calcolo fattori per la loss (importante l'ordine di ds_all)
            factors = []
            for i in range(len(single_ds_labeled)):
                factors.append([-1 * class_prior / n_labeled, 0.])
            for i in range(len(single_ds_unlabeled)):
                factors.append([0., 1 / n_unlabeled])
            factors = np.array(factors).reshape((len(factors), 2))

            # allenamento modello
            model = self.get_model()
            model.fit(rbf, factors, batch_size=rbf.shape[0], epochs=200, shuffle=True, verbose=0)

            models[k] = model

        # TESTING FINALE
        # predict labels (test and train)
        y_pred_test = self.predict(models, self.get_rbf(centers_radial, x_val, sigma))
        y_pred_train = self.predict(models, self.get_rbf(centers_radial, ds_all, sigma))

        # get accuracy
        test_mes = self.get_accuracy(y_pred_test, y_val)
        train_mes = self.get_accuracy(y_pred_train, y_all)

        return train_mes, test_mes

    def get_rbf(self, centers, x, sigma):
        r = tf.reduce_sum(x * x, 1)
        r = tf.reshape(r, [-1, 1])

        rr = tf.reduce_sum(centers * centers, 1)
        rr = tf.reshape(rr, [-1, 1])

        rbf = r - 2 * tf.matmul(x, tf.transpose(centers)) + tf.transpose(rr)
        rbf = K.exp(-1 * rbf / (2 * sigma)).numpy()

        return rbf

    def predict(self, models, rbf):

        predictions = []

        for k in sorted(self.classes):
            if k in self.positive_classes:
                pred = models[k].predict(rbf)
                predictions.append(pred)
            else:
                predictions.append(np.zeros((len(rbf)), dtype='float32')) # classe negativa

        predictions = np.column_stack(predictions)
        predictions = np.argmax(predictions, axis=1)

        return predictions


class NNPU(bc.BaseClassifier):

    def the_loss(self, y_true, y_pred):

        if self.type == "nnpu":
            loss_func = (lambda z: 1. / (1. + tf.exp(-z))) #sigmoid loss

            # calcolo (y true servono solo come fattori)
            labeled_loss = tf.reduce_sum(y_true[:, 0] * loss_func(y_pred))

            unlabeled_loss = tf.reduce_sum((y_true[:, 1] - y_true[:, 0]) * loss_func(-y_pred))

            loss = tf.cond(unlabeled_loss >= 0,
                           lambda: labeled_loss + unlabeled_loss,
                           lambda: -unlabeled_loss)

            return loss
        else:
            loss_func = (lambda z: tf.math.log(1. + tf.exp(-z))) #logistic loss

            # calcolo (y true servono solo come fattori)
            labeled_loss = -1 * tf.reduce_sum(y_true[:, 0] * y_pred)

            unlabeled_loss = tf.reduce_sum(y_true[:, 1] * loss_func(-y_pred))

            return labeled_loss + unlabeled_loss #unbiased risk estimator

    def get_model(self):
        dims = [1]

        input = Input(shape=(self.input_dim,))
        l = input

        for d in dims:
            act = 'relu' if d != dims[-1] else None
            l = Dense(d, activation=act,
                      kernel_regularizer=keras.regularizers.l2(self.hyper_parameters['weight_decay']), )(l)

        model = Model(input, l)
        model.compile(Adam(self.hyper_parameters['learning_rate']), self.the_loss)

        return model

    def single_run(self, current_run):
        # dataset
        ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_val, y_val = self.get_dataset()

        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        # ottenimento prob. a priori per le classi positive (todo ora vengono lette dal dataset)
        positive_priors = dict()
        for p_c in self.positive_classes:
            els_class, _ = get_data.filter_ds(ds_all, y_all, [p_c])

            prior = len(els_class) / len(ds_all)
            positive_priors[p_c] = prior

        self.hyper_parameters['weight_decay'] = 0.005
        self.hyper_parameters['learning_rate'] = 1e-4

        # modelli
        models = dict()

        # allenare dataset per k-1 classi
        for k in self.positive_classes:
            #print("K positive=", k)

            # esempi etichettati della k-esima classe
            single_ds_labeled, _ = get_data.filter_ds(ds_labeled, y_labeled, [k])

            # esempi non etichettati
            single_ds_unlabeled, _ = get_data.filter_ds(ds_labeled, y_labeled, [c for c in self.positive_classes if c != k])
            single_ds_unlabeled = np.concatenate((single_ds_unlabeled, ds_unlabeled), axis=0)

            single_ds_all = np.concatenate((single_ds_labeled, single_ds_unlabeled), axis=0)

            #
            class_prior = positive_priors[k]
            #class_prior = len(get_data.filter_ds(ds_unlabeled, y_unlabeled, [k])[0]) / len(single_ds_unlabeled)

            n_unlabeled = len(single_ds_unlabeled)
            n_labeled = len(single_ds_labeled)

            # calcolo fattori per la loss (importante l'ordine di ds_all)
            factors = []
            for i in range(len(single_ds_labeled)):
                factors.append([class_prior / n_labeled, 0.])
            for i in range(len(single_ds_unlabeled)):
                factors.append([0., 1. / n_unlabeled])
            factors = np.array(factors).reshape((len(factors), 2))

            # allenamento modello
            model = self.get_model()
            model.fit(single_ds_all, factors, batch_size=256, epochs=200, shuffle=True, verbose=0)

            models[k] = model

        # TESTING FINALE
        # predict labels (test and train)
        y_pred_test = self.predict(models, x_val)
        y_pred_train = self.predict(models, ds_all)

        # get accuracy
        test_mes = self.get_accuracy(y_pred_test, y_val)
        train_mes = self.get_accuracy(y_pred_train, y_all)

        return train_mes, test_mes

    def predict(self, models, rbf):

        predictions = []

        for k in sorted(self.classes):
            if k in self.positive_classes:
                pred = models[k].predict(rbf)
                predictions.append(pred)
            else:
                predictions.append(np.zeros((len(rbf)), dtype='float32')) # classe negativa

        predictions = np.column_stack(predictions)
        predictions = np.argmax(predictions, axis=1)

        return predictions


if __name__ == '__main__':

    n_runs = 3
    perc_ds = 1
    perc_labeled = 0.5
    data_preparation = '01'

    for data_preparation in ['01', 'z_norm']:
        print("\n\nDATA PREPARATION:", data_preparation)
        for dataset_name in ["optdigits", 'pendigits', 'semeion', 'har', 'usps']:
            print("\n\n Dataset:", dataset_name)

            for name in ["nnpu", "upu", 'linearSVN', 'area', 'urea']:
                # get model
                if name == "linearSVM":
                    model = LinearSVM(n_runs, dataset_name, perc_ds, 1, data_preparation, name)
                elif name == "area":
                    model = AREA(n_runs, dataset_name, perc_ds, perc_labeled, data_preparation, name)
                elif name == "urea":
                    model = UREA(n_runs, dataset_name, perc_ds, perc_labeled, data_preparation, name)
                elif name == "upu" or name == "nnpu":
                    model = NNPU(n_runs, dataset_name, perc_ds, perc_labeled, data_preparation, name)
                    model.set_type(name)
                else:
                    model = None

                # get measures
                train_mes, test_mes = model.train()

                mean_mes = np.mean(test_mes)
                format = "{:5.3f}"

                print(name, "-> TEST:", format.format(np.mean(test_mes)), "+-", format.format(np.std(test_mes)))
                print("TRAIN:", format.format(np.mean(train_mes)), "+-", format.format(np.std(train_mes)))
                print('')