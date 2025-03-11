
import tensorflow as tf
  
def local_fit_model():
    """Creates and returns a fresh instance of the neural network model."""
    initializer = tf.keras.initializers.RandomUniform(
        minval=-10.0,
        maxval=10.0,
        seed = None)

    cross_section_inputs = tf.keras.Input(shape=(5,), name='input_layer')

    # q_squared, x_value, t_value, phi, k_value = tf.split(cross_section_inputs, num_or_size_splits = 5, axis = 1)
    q_squared = cross_section_inputs[:, 0:1]
    x_value = cross_section_inputs[:, 1:2]
    t_value = cross_section_inputs[:, 2:3]
    phi = cross_section_inputs[:, 3:4]
    k = cross_section_inputs[:, 4:5]

    kinematics = tf.keras.layers.concatenate([q_squared, x_value, t_value])

    x1 = tf.keras.layers.Dense(480, activation="relu6", kernel_initializer=initializer)(kinematics)
    x2 = tf.keras.layers.Dense(320, activation="relu6", kernel_initializer=initializer)(x1)
    x3 = tf.keras.layers.Dense(240, activation="relu6", kernel_initializer=initializer)(x2)
    x4 = tf.keras.layers.Dense(120, activation="relu6", kernel_initializer=initializer)(x3)
    x5 = tf.keras.layers.Dense(32, activation="relu6", kernel_initializer=initializer)(x4)

    outputs = Dense(4, activation="linear", kernel_initializer = initializer, name='cff_output_layer')(x5)

    total_cross_section_inputs = tf.keras.layers.concatenate([cross_section_inputs, outputs], axis = 1)

    total_cross_section = TotalFLayer(name='TotalFLayer')(total_cross_section_inputs)
    
    tensorflow_network = tf.keras.Model(
        inputs = cross_section_inputs,
        outputs = total_cross_section,
        name = "total_cross_section_model")

    tensorflow_network.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [
                tf.keras.metrics.MeanSquaredError()
            ])
    
    return tensorflow_network   