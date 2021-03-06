import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


class CentroidLayer(keras.layers.Layer):

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.momentum = 0.9
        self.num_features = input_shape[-1]
        self.centroid0 = self.add_weight('centroid', shape=(1, self.num_features),
                                         trainable=False,
                                         initializer=keras.initializers.zeros())
        self.centroid1 = self.add_weight('centroid', shape=(1, self.num_features),
                                         trainable=False,
                                         initializer=keras.initializers.zeros())
        pass

    def call(self, inputs, training=True, **kwargs):
        x0, x1 = inputs

        if training:
            x0_mean = keras.backend.mean(x0, 0)
            x1_mean = keras.backend.mean(x1, 0)
            # centroid : moving average
            keras.backend.moving_average_update(self.centroid0, x0_mean, self.momentum)
            keras.backend.moving_average_update(self.centroid1, x1_mean, self.momentum)
            # embedding near to centroids
            self.add_loss(0.1 * keras.backend.mean((x0_mean - keras.backend.stop_gradient(self.centroid0)) ** 2.0))
            self.add_loss(0.1 * keras.backend.mean((x1_mean - keras.backend.stop_gradient(self.centroid1)) ** 2.0))
            # centers distance margin distance
            self.add_loss(0.1 * keras.backend.relu(5.0 - keras.backend.sum((x0_mean - x1_mean) ** 2.0)))

        return x0, x1


# DATA
x0 = np.random.normal(0, 1, (200, 3))
x1 = np.random.normal(1, 1, (200, 3))

# MODEL
X0 = keras.layers.Input(shape=x0.shape[1])
X1 = keras.layers.Input(shape=x1.shape[1])

encoder = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2)
])
decoder = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(x0.shape[1])
])
clayer = CentroidLayer()
z0 = encoder(X0)
z1 = encoder(X1)
z0, z1 = clayer([z0, z1], training=True)

model = keras.Model([X0, X1], [decoder(z0), decoder(z1)])
model.compile(loss='mse', optimizer=keras.optimizers.Adam(1e-3))
model.fit([x0, x1], [x0, x1], epochs=200, batch_size=32, verbose=2)

z0 = encoder.predict(x0)
z1 = encoder.predict(x1)
C0 = clayer.centroid0.numpy()
C1 = clayer.centroid1.numpy()

plt.scatter(z0[:, 0], z0[:, 1], color='red', alpha=0.2)
plt.scatter(z1[:, 0], z1[:, 1], color='green', alpha=0.2)
plt.scatter(C0[0, 0], C0[0, 1], color='red', marker='x', s=200)
plt.scatter(C1[0, 0], C1[0, 1], color='green', marker='x', s=200)
