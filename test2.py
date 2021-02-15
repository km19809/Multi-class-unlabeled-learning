import get_data, deep_clustering as dc

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

#tf.enable_eager_execution()
print(tf.__version__)
print(tf.executing_eagerly())


# classi da considerare
positive_classes = [0, 1, 2]
negative_class = 3

all_classes = positive_classes.copy()
all_classes.append(negative_class)

# percentuale degli esempi da mantenere etichettati
perc_labeled = 0.05

# Hyper parameters
encoding_dim = 32
lambda_hyp = 0.01
convergence_stop = 0.001


# restituisce i layer per il classificatore (prima versione semplice)
def get_layers_dense(input_img):
    encoded = layers.Dense(encoding_dim * 8, activation='relu', dtype='float32', kernel_regularizer=l2(lambda_hyp))(input_img)
    encoded = layers.Dense(encoding_dim * 4, activation='relu', dtype='float32', kernel_regularizer=l2(lambda_hyp))(encoded)
    encoded = layers.Dense(encoding_dim * 2, activation='relu', dtype='float32', kernel_regularizer=l2(lambda_hyp))(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu', dtype='float32', kernel_regularizer=l2(lambda_hyp))(encoded)

    return encoded


# restituisce i layer per il classificatore (versione convoluzionale)
def get_layers_convolutional(input_img):

    encoded = layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, padding='same')(input_img)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, padding='same')(encoded)
    encoded = layers.BatchNormalization()(encoded)

    encoded = layers.MaxPool2D((2, 2))(encoded)
    encoded = layers.Dropout(0.20)(encoded)

    encoded = layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same')(encoded)
    encoded = layers.BatchNormalization()(encoded)

    encoded = layers.MaxPool2D(pool_size=(2, 2))(encoded)
    encoded = layers.Dropout(0.30)(encoded)

    encoded = layers.Conv2D(128, (3, 3), activation=tf.nn.relu, padding='same')(encoded)
    encoded = layers.BatchNormalization()(encoded)

    encoded = layers.MaxPool2D(pool_size=(2, 2))(encoded)
    encoded = layers.Dropout(0.40)(encoded)

    encoded = layers.Flatten()(encoded)
    encoded = layers.Dense(200, activation=tf.nn.relu)(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.50)(encoded)

    return encoded


# restituisce i layer per l'autoencoder convoluzionale
def get_layers_autoencoder(input_img):

    # ENCODER
    e = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    e = layers.MaxPooling2D((2, 2))(e)
    e = layers.Conv2D(64, (3, 3), activation='relu')(e)
    e = layers.MaxPooling2D((2, 2))(e)
    e = layers.Conv2D(64, (3, 3), activation='relu')(e)
    l = layers.Flatten()(e)
    l = layers.Dense(49, activation='softmax')(l)

    encoder = keras.Model(input_img, l)

    # DECODER
    d = layers.Reshape((7, 7, 1))(l)
    d = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
    d = layers.BatchNormalization()(d)
    d = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
    d = layers.BatchNormalization()(d)
    d = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(d)
    d = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

    output_autoencoder = keras.Model(input_img, d)

    return encoder, output_autoencoder


# allenamento del classificatore sui soli esempi labeled
def train_classificator(x_data_train, y_data_train, x_data_test, y_data_test):

    # definizione layers
    input_img = keras.Input(shape=x_data_train[0].shape)
    encoded = get_layers_convolutional(input_img)

    # modello restituito dalla funzione, svolgerà la funzione di encoder
    last_layer_encoder = keras.Model(input_img, encoded)

    # si aggiunge un ultimo layer avente attivazione softmax
    output_classification = layers.Dense(len(positive_classes), activation="softmax", name="output")(encoded)

    # si costruisce il classificatore
    classificator = keras.Model(input_img, output_classification)
    classificator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # TRAINING (se i parametri sono stati già salvati, li si prende da file system)
    model_loaded = False
    name_file_model = "parameters/mnist_conv_model"

    try:
        classificator.load_weights(name_file_model)
        model_loaded = True
    except Exception:
        pass

    if not model_loaded:
        # per il test saranno esclusi gli esempi con etichetta negativa
        x_data_test_1, y_data_test_1 = get_data.filter_ds(x_data_test, y_data_test, positive_classes)

        # si adattano le etichette per l'allenamento
        y_data_test_1 = keras.utils.to_categorical(y_data_test_1)
        y_data_train_1 = keras.utils.to_categorical(y_data_train)

        # allenamento
        classificator.fit(x_data_train, y_data_train_1,
                        epochs=40,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_data_test_1, y_data_test_1))

        # salvataggio parametri
        classificator.save_weights(name_file_model)

    return last_layer_encoder


# allenamento dell'autoencoder su tutti gli esempi
def train_autoencoder(x_data_train, x_data_test):

    # definizione layers
    input_img = keras.Input(shape=x_data_train[0].shape)
    encoder, output_autoencoder = get_layers_autoencoder(input_img)

    # autoencoder allenato tramite il mean squared error
    output_autoencoder.compile(optimizer="adam", loss="mse")

    # TRAINING (se è stato già allenato, si prendono i parametri da file system)
    model_loaded = False
    name_file_model = "parameters/mnist_autoenc_model"

    try:
        output_autoencoder.load_weights(name_file_model)
        model_loaded = True
    except Exception:
        pass

    if not model_loaded:
        # allenamento
        output_autoencoder.fit(x_data_train, x_data_train,
                        epochs=17,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_data_test, x_data_test))

        # salvataggio parametri
        output_autoencoder.save_weights(name_file_model)

    return encoder


def get_radius(encoded_data, centroid):
        # il raggio è dipende dalla distanza del più vicino e del più lontano esempio dal centroide
    max = np.max([np.linalg.norm(np.subtract(centroid, x_element)) for x_element in encoded_data])
    min = np.min([np.linalg.norm(np.subtract(centroid, x_element)) for x_element in encoded_data])

    return (3 * min + max) / 4


# restituisce i centroidi e i raggi delle classi positive
def get_positive_centroids(x_data, y_data, encoder):
    positive_centroids = []
    positive_radiuses = []

    for pos_class in positive_classes:
        # si effettua l'encoding degli esempi di una classe
        (x_pos_data, _) = get_data.filter_ds(x_data, y_data, [pos_class])
        encoded_pos_data = encoder.predict(x_pos_data)

        print("Centroid for " + str(pos_class) + " class, shape encoded=" + str(encoded_pos_data.shape))

        # si fa la media dei valori encoded, ciò è il centroide
        centroid = np.mean(encoded_pos_data, axis=0)
        radius = get_radius(encoded_pos_data, centroid)

        positive_centroids.append(centroid)
        positive_radiuses.append(radius)

    return positive_centroids, positive_radiuses


# restituisce centroide e raggio della classe dei negativi
def get_negative_centroid(x_data, centroids, radiuses, encoder):

    # encoding degli esempi
    encoded_data = encoder.predict(x_data)

    # rimozione di tutti quelli esempi che rientrano nell'ipersfera di un centroide positivo
    negative_samples = np.array([encoded_x for encoded_x in encoded_data
                                 if np.min([np.linalg.norm(np.subtract(centroids[index], encoded_x)) - radiuses[index] for index, _ in enumerate(positive_classes)]) > 0
                                 ])

    print("Centroid for " + str(negative_class) + " class, shape encoded=" + str(negative_samples.shape))

    # si fa la media
    centroid = np.mean(negative_samples, axis=0)
    radius = get_radius(negative_samples, centroid)

    return centroid, radius


def deep_clustering(x_data, y_data, x_test, y_test, initial_centroids, trained_encoder):

    # il modello restituisce l'indice della classe
    y_data_for_clustering = [all_classes.index(y) for y in y_data]

    # costruzione layer aggiuntivo basato sui centroidi
    clustering_layer = dc.ClusteringLayer(len(initial_centroids), name="clustering")\

    cl = clustering_layer(trained_encoder.output)
    clustering_model = keras.Model(inputs=trained_encoder.input, outputs=cl)

    # si impostano i centroidi per l'ultimo layer
    clustering_model.get_layer(name="clustering").set_weights([np.array(initial_centroids)])
    clustering_model.compile(optimizer=keras.optimizers.SGD(0.01, 0.9), loss='kld')

    # TRAIN MODEL
    update_interval = 50
    batch_size = 256
    n_epocs = 40
    maxiter = int(n_epocs * len(x_data) / batch_size) # numero massimo di iterazioni
    print("Max iterations: " + str(maxiter))

    # numero minimo di esempi per non far terminare il processo di apprendimento
    min_samples_converge = int(len(x_data) * convergence_stop)
    print("min_samples_converge: " + str(min_samples_converge))

    precedent_y_pred = None # etichette assegnate al passo precedente
    index = 0
    loss = -1

    for ite in range(int(maxiter)):
        print("Current iteration: " + str(ite))

        if ite % update_interval == 0:

            # stampa in 2D dei cluster
            if print_2d_data:
                plot_2d(trained_encoder.predict(x_data), y_data_for_clustering, clustering_layer.get_centroids())

            # update the auxiliary target distribution p
            q = clustering_model.predict(x_data, verbose=0)
            p = dc.target_distribution(q)

            # controllo della condizione di stop
            y_pred = q.argmax(1)
            if precedent_y_pred is not None:
                different_samples = sum(1 for index, _ in enumerate(y_pred) if y_pred[index] != precedent_y_pred[index])
                print("Different samples: " + str(different_samples))
                stop_iteration = different_samples < min_samples_converge
                if stop_iteration:
                    break

            precedent_y_pred = y_pred

            # stampa dell'accuratezza corrente
            if y_data is not None:
                acc_obj = keras.metrics.Accuracy()
                acc_obj.update_state(y_data_for_clustering, y_pred)

                acc = np.round(acc_obj.result(), 5)
                print("Current accuracy: " + str(acc))
                print("Loss is " + str(loss))

        # si allena il modello su un batch di esempi
        start_index = index * batch_size
        end_index = min((index + 1) * batch_size, x_data.shape[0])

        loss = clustering_model.train_on_batch(x=x_data[start_index:end_index], y=p[start_index:end_index])

        # aggiornamento indice del batch
        index = index + 1 if (index + 1) * batch_size <= x_data.shape[0] else 0

    return clustering_model


def plot_2d(x_data, y_data, centroids):
    COLORS = np.array([
        '#FF3333',  # red
        '#FF7216',  # orange
        '#FCD116',  # yellow
        '#0198E1',  # blue
        '#BF5FFF',  # purple
        '#4DBD33',  # green
        '#87421F'  # brown
    ])

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
    X = tsne.fit_transform(np.concatenate((x_data, centroids), axis=0))

    # scatter data points
    label_color = [COLORS[index] for index in y_data]
    plt.scatter(X[:-len(centroids), 0], X[:-len(centroids), 1], c=label_color, alpha=0.5, linewidths=0.7)

    # scatter centroids
    plt.scatter(X[-len(centroids):, 0], X[-len(centroids):, 1], marker="X", alpha=1)

    plt.show()


# parametri di meta controllo
training_method = 'autoencoder'
fai_il_baro = False
do_accuracy_test = False
flatten_input_data = False
print_2d_data = True

def main():

    # ottenimento esempi dal dataset
    x_train_labeled, y_train_labeled, \
    x_train_unlabeled, y_train_unlabeled, \
    x_test, y_test = get_data.get_data(positive_classes, negative_class, perc_labeled, flatten_input_data)

    # allenamento iniziale
    # può essere effettuato col classificatore o con l'autoencoder
    if training_method == "classificator":
        trained_encoder = train_classificator(x_train_labeled, y_train_labeled, x_test, y_test)
    elif training_method == "autoencoder":
        # allenamento con autoencoder
        trained_encoder = train_autoencoder(np.concatenate((x_train_unlabeled, x_train_labeled), axis = 0), x_test)

    # ottenimento centroidi e raggi positivi
    centroids, radiuses = get_positive_centroids(x_train_labeled, y_train_labeled, trained_encoder)

    # calcolo accuratezza basato sulla distanza
    if do_accuracy_test:
        x_train_encoded = trained_encoder.predict(x_train_labeled)
        accuracy_1 = 0.0
        for c in range(len(x_train_labeled)):

            # si prende la classe col centroide più vicino all'esempio
            index_class = np.argmin([np.linalg.norm(np.subtract(x_train_encoded[c], centroids[c_class])) for c_class, _ in enumerate(positive_classes)])
            predicted_class = positive_classes[index_class]

            if predicted_class == y_train_labeled[c]:
                accuracy_1 += 1

        print("Accuracy with cluster method based on "+ training_method + ": ")
        print(accuracy_1 * 100 / len(x_train_labeled))

    # ottenimento cluster negativo
    if fai_il_baro:
        # si prende forzatamente il centroide dei negativi anche se non etichettati
        (x_negative_unlabeled, _) = get_data.filter_ds(x_train_unlabeled, y_train_unlabeled, [negative_class])
        neg_encoded = trained_encoder.predict(x_negative_unlabeled)

        negative_centroid = np.mean(neg_encoded, axis=0)
        centroids.append(negative_centroid)
        radiuses.append(get_radius(neg_encoded, negative_centroid))
    else:
        # si determina il centroide
        negative_centroid, negative_radius = get_negative_centroid(x_train_unlabeled, centroids, radiuses, trained_encoder)
        centroids.append(negative_centroid)
        radiuses.append(negative_radius)

    # calcolo accuratezza basato sulla distanza (ora si hanno tutti i centroidi)
    if do_accuracy_test:
        # dataset degli esempi NON ETICHETTATI
        x_train_unlabeled_encoded = trained_encoder.predict(x_train_unlabeled)

        accuracy_1 = 0.0 # accuratezza sui positivi
        n_1 = 0
        accuracy_2 = 0.0 # acc sui negativi
        n_2 = 0
        for c in range(len(x_train_unlabeled)):
            index_class = np.argmin(
                [np.linalg.norm(np.subtract(x_train_unlabeled_encoded[c], centroids[c_class])) for c_class, _ in enumerate(all_classes)])
            predicted_class = all_classes[index_class]

            if y_train_unlabeled[c] == negative_class:
                # negative
                n_2 += 1
                if predicted_class == y_train_unlabeled[c]:
                    accuracy_2 += 1
            else:
                # positive
                n_1 += 1
                if predicted_class == y_train_unlabeled[c]:
                    accuracy_1 += 1

        print("Accuracy with cluster method based on "+ training_method + " (UNLABELED): ")
        print("positive: " + str(accuracy_1 * 100 / n_1))
        print("negative: " + str(accuracy_2 * 100 / n_2))

    # algoritmo di deep clustering su tutti gli esempi
    clustering_model = deep_clustering(np.concatenate((x_train_unlabeled, x_train_labeled), axis = 0),
                    np.concatenate((y_train_unlabeled, y_train_labeled), axis=0),
                    x_test, y_test,
                    centroids, trained_encoder)

    # test accuratezza (labeled e non)
    print_accuracy(x_train_labeled, y_train_labeled, "deep clustering (labeled)", clustering_model)
    print_accuracy(x_train_unlabeled, y_train_unlabeled, "deep clustering (unlabeled)", clustering_model)
    print_accuracy(x_test, y_test, "deep clustering (test)", clustering_model)


def print_accuracy(x, y, label, model):
    q = model.predict(x, verbose=0)
    y_pred = q.argmax(1)

    acc_obj = keras.metrics.Accuracy()
    acc_obj.update_state(y, y_pred)
    acc = np.round(acc_obj.result(), 5)

    print("Accuracy for " + label + ":" + str(acc))


main()