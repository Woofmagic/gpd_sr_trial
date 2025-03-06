import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def global_fit_model():
    """Creates and returns a fresh instance of the neural network model."""
    initializer = tf.keras.initializers.RandomUniform(
        minval = -10.0,
        maxval = 10.0,
        seed = None)

    input_kinematics = Input(shape=(3,), name = "input_kinematics")

    x1 = tf.keras.layers.Dense(480, activation="relu6", kernel_initializer=initializer)(input_kinematics)
    x2 = tf.keras.layers.Dense(320, activation="relu6", kernel_initializer=initializer)(x1)
    x3 = tf.keras.layers.Dense(240, activation="relu6", kernel_initializer=initializer)(x2)
    x4 = tf.keras.layers.Dense(120, activation="relu6", kernel_initializer=initializer)(x3)
    x5 = tf.keras.layers.Dense(32, activation="relu6", kernel_initializer=initializer)(x4)

    output_y_value = Dense(4, activation = "linear", kernel_initializer = initializer, name = "output_global_cff_values")(x5)

    tensorflow_network = Model(
        inputs = input_kinematics,
        outputs = output_y_value,
        name = "global_fit_cffs")

    tensorflow_network.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [
                tf.keras.metrics.MeanSquaredError()
            ])

    return tensorflow_network