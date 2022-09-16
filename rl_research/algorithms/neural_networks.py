import copy
import numpy as np
from scipy.special import softmax
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras


class np_nn_softmax_out:
    def __init__(self, inp=2, h1=256, h2=256, out=3, init_weights=None):
        if init_weights:
            self.init_weights(init_weights)
        else:
            self.weights = {
                'w1': self.xavier_init(inp, h1),
                'b1': np.zeros((1, h1)),
                'w2': self.xavier_init(h1, h2),
                'b2': np.zeros((1, h2)),
                'w3': self.xavier_init(h2, out),
                'b3': np.zeros((1, out))
            }

    @staticmethod
    def xavier_init(h1, h2):
        glorot = 1.0 * np.sqrt(6.0 / (h1 + h2))
        size = (h1, h2)
        return np.random.uniform(-glorot, glorot, size)

    @staticmethod
    def relu(l):
        return np.where(l < 0, 0, l)

    @staticmethod
    def softmax(l):
        e_x = np.exp(l - np.max(l))
        return e_x / e_x.sum(axis=-1)

    def init_weights(self, init_weights):
        self.weights = copy.deepcopy(init_weights)

    def forward(self, inp):
        w1 = self.weights['w1']
        b1 = self.weights['b1']
        w2 = self.weights['w2']
        b2 = self.weights['b2']
        w3 = self.weights['w3']
        b3 = self.weights['b3']

        l1 = self.relu(inp @ w1 + b1)
        l2 = self.relu(l1 @ w2 + b2)
        out = self.softmax(l2 @ w3 + b3)

        return out


class tf_nn_softmax_out:
    def __init__(self, *args, **kwargs):
        self.build_model(*args, **kwargs)

    def build_model(self, inp=2, h1=256, h2=256, out=3, init_weights=None):
        self.inp = inp
        self.out = out

        inp = keras.Input(shape=(inp, ))
        x = keras.layers.Dense(
            h1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(inp)
        x = keras.layers.Dense(
            h2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(x)
        outp = keras.layers.Dense(
            out, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform')(x)

        self.lr = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.95)
        # self.lr = 1e-2

        self.optimizer = keras.optimizers.SGD(learning_rate=self.lr)
        self.model = keras.Model(inputs=inp, outputs=outp)

        self.weights = self.model.trainable_weights

    def predict(self, inp):
        if not isinstance(inp, tf.Tensor):
            inp = tf.convert_to_tensor(np.array(inp).reshape(1, -1))
        return np.argmax(self.model(inp))

    def forward(self, inp):
        if not isinstance(inp, tf.Tensor):
            inp = tf.convert_to_tensor(np.array(inp).reshape(-1, self.inp))
        probs = self.model(inp)
        return probs.numpy()

    def distributions(self, inp):
        if not isinstance(inp, tf.Tensor):
            inp = tf.convert_to_tensor(np.array(inp).reshape(1, -1))
        probs = self.model(inp)
        return tfp.distributions.Categorical(probs=probs)

    def update_params(self, grads):
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))


class tf_nn_linear_out:
    def __init__(self, *args, **kwargs):
        self.build_model(*args, **kwargs)

    def build_model(self, inp=4, h1=256, h2=256, out=1, init_weights=None):
        self.inp = inp
        self.out = out

        inp = keras.Input(shape=(inp, ))
        x = keras.layers.Dense(
            h1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(inp)
        x = keras.layers.Dense(
            h2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(x)
        outp = keras.layers.Dense(
            out, activation='linear', use_bias=True, kernel_initializer='glorot_uniform')(x)

        self.lr = 1e-3
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.model = keras.Model(inputs=inp, outputs=outp)

        self.weights = self.model.trainable_weights

    def forward(self, inp):
        if not isinstance(inp, tf.Tensor):
            inp = tf.convert_to_tensor(np.array(inp).reshape(-1, self.inp))
        return self.model(inp)

    def predict(self, inp):
        if not isinstance(inp, tf.Tensor):
            inp = tf.convert_to_tensor(np.array(inp).reshape(-1, self.inp))
        return self.model(inp).numpy()

    def update_params(self, grads):
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))
