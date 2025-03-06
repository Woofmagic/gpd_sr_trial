import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import Model

def global_fit_model():
    """Creates and returns a fresh instance of the neural network model."""
    initializer = tf.keras.initializers.RandomUniform(
        minval = -10.0,
        maxval = 10.0,
        seed = None)

    circle_function_input = Input(shape=(3,), name = "spherical_function_inputs")

    x1 = tf.keras.layers.Dense(480, activation="relu6", kernel_initializer=initializer)(circle_function_input)
    x2 = tf.keras.layers.Dense(320, activation="relu6", kernel_initializer=initializer)(x1)
    x3 = tf.keras.layers.Dense(240, activation="relu6", kernel_initializer=initializer)(x2)
    x4 = tf.keras.layers.Dense(120, activation="relu6", kernel_initializer=initializer)(x3)
    x5 = tf.keras.layers.Dense(32, activation="relu6", kernel_initializer=initializer)(x4)

    output_y_value = Dense(4, activation = "linear", kernel_initializer = initializer, name = "output_radial_value")(x5)

    tensorflow_network = Model(inputs = circle_function_input, outputs = output_y_value, name = "spherical_function_fitter")

    return tensorflow_network