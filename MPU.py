# FORGET THIS FILE, MPU IS NOT EFFECTIVE


import base_classifier as bc
import numpy as np
from keras.layers import Input, Dense
from keras import Model
import keras
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import datasets as ds
import math
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class MPU(bc.BaseClassifier):

    def the_loss(self, y_true, y_pred):

        true_moltiplicator = y_true[:, 2 + len(self.classes):]
        vector_true = tf.reduce_sum(y_pred * true_moltiplicator, 1)

        # calcolo per parte labeled
        k_pred = y_pred[:, -1]  # parte k-esima predetta
        loss = y_true[:, 0] * tf.maximum(0., k_pred - vector_true)
        loss = tf.reduce_sum(loss)

        # parte unlabeled
        loss_unlab = tf.transpose(y_pred[:, :-1])  # parte predetta per le label
        loss_unlab = loss_unlab - vector_true
        loss_unlab = y_true[:, 1] * tf.maximum(0., 1. + loss_unlab)

        loss += tf.reduce_sum(loss_unlab)

        return loss

    def get_encoding_labels(self):
        labels = []

        k = len(self.classes)
        r = k - 1

        a = (1 + math.sqrt(r + 1)) / r
        center = (a + 1) / (r + 1)

        for v in range(k - 1):
            vector = []
            for p in range(r):
                point = (0 if p != v else 1) - center
                vector.append(point)

            labels.append(vector)

        labels.append([a - center for _ in range(r)])

        # scale data for unit distance from origin
        scale = math.sqrt(sum([d ** 2 for d in labels[0]]))

        # todo experimental
        scale /= 1

        # rotazione (experimental)
        theta = np.radians(225)
        for i in range(r - 1):
            rot_matrix = np.identity(r, dtype='float64')
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))

            rot_matrix[i:i+2, i:i+2] = R

            for i in range(len(labels)):
                labels[i] = np.matmul(labels[i], rot_matrix)

        labels = np.array(labels) / scale

        # check proprietà
        #for l in labels:
        #    print(np.sum(l ** 2))
        #for i in range(len(labels)):
        #    for j in range(len(labels)):
        #        if i != j:
        #            print(np.sum((labels[i] - labels[j]) ** 2))

        return labels

    def get_pseudolabels(self, ds_unlabeled, model):

        #return [np.argmax(pred) for pred in model.predict(ds_unlabeled)]

        return [np.argmin([sum([max(0., 1. + p - pred[y]) for p in pred[:-1]]) for y in range(len(self.classes))]) for pred in model.predict(ds_unlabeled)]

        predictions = model.predict(ds_unlabeled)

        pseudo_labels = []
        for pred in predictions:
            min_y = None
            best_loss = None
            for y in range(len(self.classes)):

                loss = sum([max(0., 1. + p - pred[y]) for p in pred[:-1]])

                if best_loss is None or loss < best_loss:
                    min_y = y
                    best_loss = loss

            pseudo_labels.append(min_y)

        return np.array(pseudo_labels)

    def get_grid_hyperparameters(self):
        if self.validate_hyp:
            return {
                'Learning_rate': np.logspace(-4, 0, 6),
                #'Weight_decay': np.logspace(-5, -5, 1),
                'epochs': np.logspace(0, 3, 4)
            }
        else:
            return {
                'Learning_rate': [1e-2],
                #'Weight_decay': [1e-5],
                'epochs': [10]
            }

    def get_model(self, input_dim, hyp):
        # I bias non vengono utilizzati
        weight_decay = 1e-5

        dims = [len(self.classes) - 1, len(self.classes)]

        input = Input(shape=(input_dim,))
        l = input

        for i in range(len(dims)):
            act = 'relu' if i < len(dims) - 2 else 'linear'
            trainable = i < len(dims) - 1

            init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
            init = 'glorot_uniform'

            l = Dense(dims[i], activation=act, use_bias=False, trainable=trainable,  kernel_initializer=init,
                      kernel_regularizer=keras.regularizers.l2(weight_decay),)(l)

        model = Model(input, l)
        model.compile(Adam(hyp['Learning_rate']), self.the_loss, metrics=[self.accuracy_metric])

        return model

    def predict(self, model, x):
        return np.argmax(model.predict(x), axis=1)

    def accuracy_metric(self, y_true, y_pred):
        #y_true_num = tf.argmax(y_true[:, 2:2 + len(self.classes)], axis=-1)
        #y_pred_num = tf.argmax(y_pred, axis=-1)

        #return tf.py_function(func=self.get_accuracy, inp=[y_pred_num, y_true_num], Tout=[tf.float32])

        return tf.metrics.categorical_accuracy(y_true[:, 2:2 + len(self.classes)], y_pred)

    @staticmethod
    def get_accuracy(y_pred, y_true):

        acc = sum([y_pred[i] == y_true[i] for i, _ in enumerate(y_pred)]) / len(y_pred)

        # cluster accuracy
        '''y_true1 = np.array(y_true).astype(np.int64)
        y_pred = np.array(y_pred)

        D = max(y_pred.max(), y_true1.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true1[i]] += 1

        row, col = linear_assignment(w.max() - w)

        acc = sum([a for a in w[row, col]]) * 1.0 / y_pred.size
'''
        return acc

    def train_model(self, model, ds_labeled, y_labeled, ds_unlabeled, y_unlabeled, x_test, y_test, current_hyp):

        K = len(self.classes)

        # determinazione dei fattori da utilizzare per la loss
        ds_all = np.concatenate((ds_unlabeled, ds_labeled), axis=0)
        y_all = np.concatenate((y_unlabeled, y_labeled), axis=0)

        # labels utilizzate per la validazione
        y_all_categorical = keras.utils.to_categorical(y_all, len(self.classes))
        y_test_categorical = keras.utils.to_categorical(y_test, len(self.classes))
        categ_y_test = np.array([[0, 0, ] + [_y for _y in y] + [0 for _ in range(K)] for y in y_test_categorical])

        # Determinazione pesi per l'ultimo layer (embeddings)
        embeddings = self.get_encoding_labels()

        '''model_p = Model(model.input, model.layers[-2].output)

        #unlabeled
        y_lab = model_p.predict(ds_unlabeled)
        y_lab = tf.matmul(y_lab, tf.cast(embeddings.transpose(), 'float32'))

        mean = np.mean(y_lab, axis=0)
        argmin = np.argmax(mean)

        emb_swap = embeddings[-1].copy()
        embeddings[-1] = embeddings[argmin].copy()
        embeddings[argmin] = emb_swap.copy()

        for y in sorted(self.positive_classes):
            y_lab, _ = ds.filter_ds(ds_labeled, y_labeled, [y])

            y_lab = model_p.predict(y_lab)
            y_lab = tf.matmul(y_lab, tf.cast(embeddings.transpose(), 'float32'))

            mean = np.mean(y_lab, axis=0)
            argmin = np.argmax([z if y <= i < len(self.positive_classes) else -999999. for i, z in enumerate(mean)])

            #mean = np.mean(model_p.predict(y_lab), axis=0)
            #argmin = np.argmin([sum((mean - emb) ** 2) if i >= y else 9999 for i, emb in enumerate(embeddings)])

            if y != argmin:
                emb_swap = embeddings[y].copy()
                embeddings[y] = embeddings[argmin].copy()
                embeddings[argmin] = emb_swap.copy()

            print(argmin)

        w_last = model.layers[-1].get_weights()
        w_last[0] = embeddings.transpose()
        model.layers[-1].set_weights(w_last)

        #print("Embeddings")
        #print(embeddings)'''
        # fine lavoro su embeddings

        # determinazione dei fattori da utilizzare per la loss
        N_U = len(ds_unlabeled)

        # ottenimento probabilità a priori (per ora vengono lette dal dataset)
        positive_class_factors = []
        for p_c in self.positive_classes:
            els_class, _ = ds.filter_ds(ds_all, y_all, [p_c])
            els_class_labeled, _ = ds.filter_ds(ds_labeled, y_labeled, [p_c])

            prior = len(els_class) / len(ds_all)
            n_labeled = len(els_class_labeled)

            positive_class_factors.append(prior / n_labeled)

        product_loss_lab = []
        product_loss_unlab = []

        # fattori per gli esempi non etichettati (è importante come viene definito ds_all)
        for i in range(N_U):
            product_loss_lab.append(0.)
            product_loss_unlab.append(1 / (2 * N_U * (K - 1)))

        # fattori per gli esempi etichettati (vedere la definizione nel paper)
        for y in y_labeled:
            product_loss_lab.append(positive_class_factors[y] / (2 * (K - 1)))
            product_loss_unlab.append(0.)

        tol = 0.001
        epochs_per_iter = int(current_hyp['epochs'])
        max_iter = 300

        old_pseudo_y_unlab = None
        iter = 0
        history = None
        pretraining_epochs = 0
        pretraining_iters = 500


        #
        eps = 1e-7
        l_pen = 0.00001
        init = tf.keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001)
        #init = tf.keras.initializers.RandomNormal(stddev=1 / np.std(ds_all))
        #init = tf.keras.initializers.GlorotUniform(0)
        #init = tf.keras.initializers.Orthogonal()

        W = tf.Variable(init((len(ds_all[0]), len(self.classes) - 1)))
        embeddings_t = tf.constant(embeddings, dtype='float32')





        ##########################################

        min_W = W.numpy()
        min_embeddings = np.array(embeddings)

        yss = keras.utils.to_categorical([y for y in range(len(self.classes))])
        eps = 1e-6

        def min_loss(_W, x):
            y_pred = np.matmul(np.matmul(x, _W), np.transpose(min_embeddings))

            vector_true = np.sum(y_pred * yss, 1)

            loss_unlab = np.transpose(y_pred[:, :-1])
            loss_unlab = loss_unlab - vector_true
            loss_unlab = np.maximum(0., 1. + loss_unlab)

            res = np.sum(loss_unlab, axis=0)
            return res

        # fattori per la loss
        min_products = np.concatenate((
            [positive_class_factors[y] / (2 * (K - 1)) for y in y_labeled],
            [1 / ((K - 1) * 2 * N_U) for _ in range(N_U * (K - 1))]), axis=0)

        min_arg_sum = np.concatenate((
            [0. for y in y_labeled],
            [1 for _ in range(N_U * (K - 1))]), axis=0)

        min_arg_cod_pos = np.concatenate((
            [embeddings[-1] for y in y_labeled],
            [embeddings[i // N_U] for i in range(N_U * (K - 1))]), axis=0)

        min_arg_cod_neg = np.concatenate((
            [embeddings[y] for y in y_labeled],
            [np.empty((self.num_classes - 1)) for _ in range(N_U * (K - 1))]), axis=0)

        min_x = np.concatenate((
            ds_labeled,
            [ds_unlabeled[i % N_U] for i in range(N_U * (K - 1))]), axis=0)

        def from_w(__W):
            return np.reshape(__W, min_W.shape)

        def func_to_minimize(__W):

            _W = from_w(__W)
            min_mult = np.matmul(min_x, _W)

            positive_prod = np.sum(min_mult * min_arg_cod_pos, axis=1)
            negative_prod = np.sum(min_mult * min_arg_cod_neg, axis=1)

            hinge_argument = min_arg_sum + positive_prod - negative_prod
            hinge = np.maximum(0., hinge_argument)

            loss = hinge * min_products

            frob = _W ** 2
            frob = np.sum(frob)

            res = np.sum(loss)

            return res + frob * l_pen

        def func_to_minimize_tensor(_W):

            min_mult = tf.matmul(tf.cast(min_x, dtype='float64'), _W)

            positive_prod = tf.reduce_sum(min_mult * min_arg_cod_pos, axis=1)
            negative_prod = tf.reduce_sum(min_mult * min_arg_cod_neg, axis=1)

            hinge_argument = min_arg_sum + positive_prod - negative_prod
            hinge = tf.maximum(0., hinge_argument)

            loss = hinge * min_products

            frob = _W ** 2
            frob = tf.reduce_sum(frob)

            res = tf.reduce_sum(loss)

            return res + frob * l_pen

        def jac_fun(__W):
            _W = tf.Variable(from_w(__W))
            with tf.GradientTape() as tape:
                loss = func_to_minimize_tensor(_W)

            grads = tape.gradient(loss, [_W])

            res = grads[0].numpy().flatten()

            return res

        old_loss = 0
        for i in range(100):
            print("Iter", i)

            # get argmin labels
            pseudo_y_unlab = [np.argmin(min_loss(min_W, [x])) for x in ds_unlabeled]

            # accuracy
            print("UNLAB ACC", self.get_accuracy(y_unlabeled, pseudo_y_unlab))

            predicted_test = np.matmul(np.matmul(x_test, min_W), np.transpose(min_embeddings))
            predicted_test = np.argmax(predicted_test, axis=1)
            print("TEST ACC", self.get_accuracy(y_test, predicted_test))

            # get correct parameters
            unlabeled_zed = [embeddings[y] for y in pseudo_y_unlab]
            unlabeled_zed = np.tile(unlabeled_zed, (K - 1, 1))

            min_arg_cod_neg[len(ds_labeled):] = unlabeled_zed

            # func minimization
            res = minimize(func_to_minimize, min_W, method='Newton-CG', jac=jac_fun, tol=eps)
            min_W = from_w(res.x)

            if abs(old_loss - res.fun) < eps:
                break
            else:
                old_loss = res.fun
                print(old_loss)

        ####################

        print("FINISH")
        a = 43 / 0




        opt = tf.keras.optimizers.Adam(learning_rate=0.001, )

        def get_the_loss(product_loss_unlab_, product_loss_lab_, for_pseudo=False):
            def losss(_W, y, x):

                y_pred = tf.matmul(tf.matmul(x, _W), embeddings_t, transpose_b=True)

                vector_true = tf.reduce_sum(y_pred * y, 1)

                # parte unlabeled
                loss_unlab = tf.transpose(y_pred[:, :-1])  # parte predetta per le label
                loss_unlab = loss_unlab - vector_true
                loss_unlab = product_loss_unlab_ * tf.maximum(0., 1. + loss_unlab)

                if for_pseudo:
                    res = tf.reduce_sum(loss_unlab, axis=0)
                    return res

                # calcolo per parte labeled
                k_pred = y_pred[:, -1]  # parte k-esima predetta
                loss_lab = product_loss_lab_ * tf.maximum(0., k_pred - vector_true)

                # frobenius norm
                w_loss = _W ** 2
                w_loss = tf.reduce_sum(w_loss)

                loss = tf.reduce_sum(loss_lab) + tf.reduce_sum(loss_unlab) + l_pen * w_loss
                return loss

            return losss

        def get_the_loss_pretraining():

            cce = tf.keras.losses.CategoricalCrossentropy()

            def losss(_W, y, x):
                y_pred = tf.matmul(tf.matmul(x, _W), embeddings_t, transpose_b=True)
                y_pred = tf.keras.activations.softmax(y_pred, axis=-1)

                loss =  cce(y_pred, y)
                return loss

            return losss
        #

        the_losss = get_the_loss(product_loss_unlab, product_loss_lab)

        the_losss_pre = get_the_loss(product_loss_unlab[N_U:], product_loss_lab[N_U:])
        #the_losss_pre = get_the_loss_pretraining()

        da_loss = get_the_loss([1.], [0.], True)
        yss = tf.constant(keras.utils.to_categorical([y for y in range(len(self.classes))]))
        old_loss_prev = 0

        for i in range(100):
            print("Iter", i)

            # get argmin labels
            pseudo_y_unlab = [tf.argmin(da_loss(W, yss, [x])).numpy() for x in ds_unlabeled]


            #pseudo_y_unlab = []
            #for i in range(len(ds_unlabeled)):
            #    v = da_loss(W, yss, [ds_unlabeled[i]])
            #    min_y = tf.argmin(v)
            #    pseudo_y_unlab.append(min_y.numpy())

            # convergence criterium
            '''if old_pseudo_y_unlab is not None:
                delta_label = sum(pseudo_y_unlab[i] != old_pseudo_y_unlab[i] for i in range(len(pseudo_y_unlab))) / len(pseudo_y_unlab)
                if delta_label <= tol:
                    print('Reached stopping criterium, delta_label ', delta_label, '< tol ', tol, ". Iter n°", iter)
                    break
            old_pseudo_y_unlab = pseudo_y_unlab.copy()'''

            if iter < pretraining_epochs:
                pseudo_y_all = np.concatenate(([self.num_classes - 1 for _ in range(len(ds_unlabeled))], y_labeled), axis=0)
                ycat = tf.constant(keras.utils.to_categorical(pseudo_y_all, len(self.classes)))
            else:
                pseudo_y_all = np.concatenate((pseudo_y_unlab, y_labeled), axis=0)

            old_loss = 0
            iiii = 0

            while True:
                iiii += 1

                with tf.GradientTape() as tape:

                    if iter < pretraining_epochs:
                        #loss = the_losss_pre(W, ycat, ds_all)
                        loss = the_losss_pre(W, ycat[N_U:], ds_all[N_U:])
                    else:
                        ycat = tf.constant(keras.utils.to_categorical(pseudo_y_all, len(self.classes)))
                        loss = the_losss(W, ycat, ds_all)

                # fermati se la loss non diminuisce piu
                if iter < pretraining_epochs:
                    if iiii >= pretraining_iters:
                        break
                else:
                    if abs(loss - old_loss) <= eps:
                        break

                old_loss = loss

                if iiii % 1 == 0:
                    print(loss.numpy())

                grads = tape.gradient(loss, [W])
                processed_grads = [g for g in grads]
                grads_and_vars = zip(processed_grads, [W])
                opt.apply_gradients(grads_and_vars)

            print("UNLAB ACC", self.get_accuracy(y_unlabeled, pseudo_y_unlab))

            predicted_test = tf.matmul(tf.matmul(x_test, W), embeddings_t, transpose_b=True)
            predicted_test = tf.argmax(predicted_test, axis=1).numpy()

            print("TEST ACC", self.get_accuracy(y_test, predicted_test))

            # fermati se la loss non diminuisce piu
            if iter >= pretraining_epochs:
                if abs(old_loss_prev - old_loss) <= eps:
                    break
                old_loss_prev = old_loss


        if False:
            while iter < max_iter:
                #print(iter)
                if iter % 10 == -1:

                    means = []
                    for y in self.classes:
                        dsx, _ = ds.filter_ds(ds_all, y_all, [y])
                        dsy = model_p.predict(dsx)
                        mean = np.mean(dsy, axis=0)
                        #print("[{}] -> {}".format(y, mean))
                        means.append(mean)

                    allc = np.concatenate((embeddings, means), axis=0)
                    self.plot_clusters2(model, ds_all, y_all, np.array([i >= N_U for i in range(len(ds_all))]),  allc, iter)

                # get argmax labels
                pseudo_y_unlab = self.get_pseudolabels(ds_unlabeled, model)

                # convergence criterium
                if old_pseudo_y_unlab is not None and iter > pretraining_epochs:
                    delta_label = sum(pseudo_y_unlab[i] != old_pseudo_y_unlab[i] for i in range(len(pseudo_y_unlab))) / len(pseudo_y_unlab)
                    if delta_label < tol:
                        print('Reached stopping criterium, delta_label ', delta_label, '< tol ', tol, ". Iter n°", iter)
                        break

                old_pseudo_y_unlab = pseudo_y_unlab.copy()

                pseudo_y_all = np.concatenate((pseudo_y_unlab, y_labeled), axis=0)

                #for y in range(len(self.classes)):
                #    asd = sum([y == pseudo_y_all[i] for i in range(len(ds_all))])
                #    print("y:", y, " -> ", asd)
                print("ACC", self.get_accuracy(y_all, pseudo_y_all))


                # si calcolano i fattori da dare in pasto alla loss
                factors = [[product_loss_lab[i]] + [product_loss_unlab[i]] + [x for x in y_all_categorical[i]] + [0. if k != pseudo_y_all[i] else 1. for k in range(K)]
                #factors = [[product_loss_lab[i]] + [product_loss_unlab[i]] + [x for x in y_all_categorical[i]] + [x for x in y_all_categorical[i]]
                    for i in range(len(ds_all))]

                factors2 = [[product_loss_lab[i]] + [product_loss_unlab[i]] + [x for x in y_all_categorical[i]] + [x for x in y_all_categorical[i]]
                           for i in range(len(ds_all))]

                asd = 0
                for i in range(len(ds_all)):
                    #if i < N_U:
                    #    continue
                    asd1 = factors2[i][2+K:]
                    asd2 = factors[i][2+K:]
                    boo = sum([abs(asd1[j] - asd2[j]) for j in range(len(asd1))]) > 0
                    if boo:
                        asd += 1

                print("asd", asd)

                factors = np.array(factors)

                # train parameters
                if iter > pretraining_epochs:
                    _history = model.fit(ds_all, factors, batch_size=256, epochs=epochs_per_iter, shuffle=True,
                                  validation_data=(x_test, categ_y_test), verbose=0)
                else:
                    _history = model.fit(ds_all[N_U:], factors[N_U:], batch_size=256, epochs=epochs_per_iter, shuffle=True,
                                         validation_data=(x_test, categ_y_test), verbose=0)

                # si mantiene la storia di tutte le epoche
                if history is None:
                    history = _history
                else:
                    history.epoch.extend([x + iter * epochs_per_iter for x in _history.epoch])
                    for key in _history.history.keys():
                        value = _history.history[key]
                        history.history[key].extend(value)

                iter += 1

        means = []
        for y in self.classes:
            dsx, _ = ds.filter_ds(ds_all, y_all, [y])
            dsy = model_p.predict(dsx)
            mean = np.mean(dsy, axis=0)
            #print("[{}] -> {}".format(y, mean))
            means.append(mean)

        allc = np.concatenate((embeddings, means), axis=0)
        self.plot_clusters2(model, ds_all, y_all, np.array([i >= N_U for i in range(len(ds_all))]), allc, iter)



        #a = self.predict(model, ds_all)
        #for y in range(len(self.classes)):
        #    asd = sum([y == a[i] for i in range(len(ds_all))])
        #    print("y:", y, " -> ", asd)

        return history

    def plot_clusters2(self, model, x, y_true, index_labeled, centroids, iter):

        perc_to_show = 1

        # si prende una parte dei dati (usato per velocizzare)
        shuffler1 = np.random.permutation(len(x))
        indexes_to_take = np.array([t for i, t in enumerate(shuffler1) if i < len(x) * perc_to_show])
        labeled_for_tsne = index_labeled[indexes_to_take]

        y_pred = self.predict(model, x)

        model_p = Model(model.input, model.layers[-2].output)

        data_for_tsne = model_p.predict(x[indexes_to_take])
        data_for_tsne = np.concatenate((data_for_tsne, centroids), axis=0)

        # get data in 2D (include centroids)
        x_embedded = TSNE(n_components=2, verbose=0).fit_transform(data_for_tsne)

        vis_x = x_embedded[:-len(centroids), 0]
        vis_y = x_embedded[:-len(centroids), 1]

        labeled_samples_x = np.array([x for i, x in enumerate(vis_x) if labeled_for_tsne[i]])
        labeled_samples_y = np.array([x for i, x in enumerate(vis_y) if labeled_for_tsne[i]])

        # 4 figures
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('Predicted vs True | All vs Labeled')
        cmap = plt.cm.get_cmap("jet", 256)

        # PREDICTED
        if y_pred is not None:
            # all
            y_pred_for_tsne = y_pred[indexes_to_take]
            ax1.scatter(vis_x, vis_y, c=y_pred_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.3)

            # labeled
            labeled_y_for_tsne = np.array([x for i, x in enumerate(y_pred_for_tsne) if labeled_for_tsne[i]])
            ax3.scatter(labeled_samples_x, labeled_samples_y, c=labeled_y_for_tsne, linewidths=0.2,
                        marker=".",
                        cmap=cmap, alpha=0.5)

        # TRUE

        # all
        y_true_for_tsne = y_true[indexes_to_take]
        ax2.scatter(vis_x, vis_y, c=y_true_for_tsne, linewidths=0.2, marker=".", cmap=cmap, alpha=0.3)

        # labeled
        labeled_y_true_for_tsne = np.array(
            [x for i, x in enumerate(y_true_for_tsne) if labeled_for_tsne[i]])
        ax4.scatter(labeled_samples_x, labeled_samples_y, c=labeled_y_true_for_tsne, linewidths=0.2,
                    marker=".",
                    cmap=cmap, alpha=0.5)

        # CENTROIDS
        if len(centroids):
            label_color = [index for index, _ in enumerate(centroids)]
            ax1.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X",
                        alpha=1,
                        c=label_color,
                        edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

            ax2.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X",
                        alpha=1,
                        c=label_color,
                        edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

            ax3.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X",
                        alpha=1,
                        c=label_color,
                        edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

            ax4.scatter(x_embedded[-len(centroids):, 0], x_embedded[-len(centroids):, 1], marker="X",
                        alpha=1,
                        c=label_color,
                        edgecolors="#FFFFFF", linewidths=1, cmap=cmap)

        # color bar
        norm = plt.cm.colors.Normalize(vmax=len(self.classes) - 1, vmin=0)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3)

        path = self.path_for_files + "Clusters2D" + str(iter) + ".jpg"
        plt.savefig(path)
        plt.close(fig)
