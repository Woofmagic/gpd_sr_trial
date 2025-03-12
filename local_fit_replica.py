
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import split_data

SETTING_VERBOSE = True
SETTING_DEBUG = True
Learning_Rate = 0.001
EPOCHS = 3000
BATCH_SIZE_LOCAL_FITS = 25
BATCH_SIZE_GLOBAL_FITS = 10
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9
SETTING_DNN_TRAINING_VERBOSE = 1
_TRAIN_VALIDATION_PERCENTAGE = 0.10

def generate_replica_data(
        pandas_dataframe: pd.DataFrame,
        mean_value_column_name: str,
        stddev_column_name: str,
        new_column_name: str):
    """Generates a replica dataset by sampling F within sigmaF."""
    pseudodata_dataframe = pandas_dataframe.copy()

    # Ensure error values are positive
    pseudodata_dataframe[stddev_column_name] = np.abs(pandas_dataframe[stddev_column_name])

    # Generate normally distributed F values
    replica_cross_section_sample = np.random.normal(
        loc = pandas_dataframe[mean_value_column_name], 
        scale = pseudodata_dataframe[stddev_column_name])

    # Prevent negative values (ensuring no infinite loops)
    pseudodata_dataframe[mean_value_column_name] = np.maximum(replica_cross_section_sample, 0)

    # Store original F values
    pseudodata_dataframe[new_column_name] = pandas_dataframe[mean_value_column_name]

    return pseudodata_dataframe

def run_local_fit_replica_method(
        number_of_replicas,
        model_builder,
        data_file,
        kinematic_set_number):

    if SETTING_VERBOSE:
        print(f"> Beginning Replica Method with {number_of_replicas} total Replicas...")

    this_replica_data_set = data_file[data_file['set'] == kinematic_set_number].reset_index(drop = True)
    this_replica_data_set.to_csv(f"experimental_data_kinematic_set_{kinematic_set_number}.csv")

    for replica_index in range(number_of_replicas):

        if SETTING_VERBOSE:
            print(f"> Now initializing replica #{replica_index + 1}...")

        tensorflow_network = model_builder()

        if SETTING_DEBUG:
            print("> Successfully initialized a TensorFlow network!")

        pseudodata_dataframe = generate_replica_data(
            pandas_dataframe = this_replica_data_set,
            mean_value_column_name = 'F',
            stddev_column_name = 'sigmaF',
            new_column_name = 'True_F')
        
        pseudodata_dataframe.to_csv(f"pseudodata_kinematic_set_{kinematic_set_number}_replica_{replica_index+1}.csv")
        
        if SETTING_DEBUG:
            print(f"> Successfully generated pseudodata dataframe for replica #{replica_index+1} in kinematic set {kinematic_set_number}")

        training_x_data, testing_x_data, training_y_data, testing_y_data, training_y_error, testing_y_error = split_data(
            x_data = pseudodata_dataframe[['QQ', 'x_b', 't', 'phi_x', 'k']],
            y_data = pseudodata_dataframe['F'],
            y_error_data = pseudodata_dataframe['sigmaF'],
            split_percentage = _TRAIN_VALIDATION_PERCENTAGE)
        
        if SETTING_DEBUG:
            print(f"> Successfully split pseudodata into training and testing data with a train/validation split of {_TRAIN_VALIDATION_PERCENTAGE}")
        
        # (1): Set up the Figure instance
        figure_instance_predictions = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance_predictions = figure_instance_predictions.add_subplot(1, 1, 1)
        
        plot_customization_predictions = PlotCustomizer(
            axis_instance_predictions,
            title = r"$\sigma$ vs. $\phi$ for Replica {} at Kinematic Setting {}".format(replica_index+1, kinematic_set_number),
            xlabel = r"$\phi$ [deg]",
            ylabel = r"$\sigma$ [$nb/GeV^{4}$]")
        
        plot_customization_predictions.add_errorbar_plot(
            x_data = this_replica_data_set['phi_x'],
            y_data = this_replica_data_set['F'],
            x_errorbars = np.zeros(this_replica_data_set['sigmaF'].shape),
            y_errorbars = this_replica_data_set['sigmaF'],
            label = r'Raw Data',
            color = "black",)
        
        plot_customization_predictions.add_errorbar_plot(
            x_data = pseudodata_dataframe['phi_x'],
            y_data = pseudodata_dataframe['F'],
            x_errorbars = np.zeros(pseudodata_dataframe['sigmaF'].shape),
            y_errorbars = np.zeros(pseudodata_dataframe['sigmaF'].shape),
            label = r'Generated Pseudodata',
            color = "red",)
        
        figure_instance_predictions.savefig(f"cross_section_vs_phi_kinematic_set_{kinematic_set_number}_replica_{replica_index+1}.png")

        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        print(f"> Replica #{replica_index + 1} now running...")

        history_of_training = tensorflow_network.fit(
            training_x_data,
            training_y_data,
            validation_data = (testing_x_data, testing_y_data),
            epochs = EPOCHS,
            # callbacks = [
            #     tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = modify_LR_factor, patience = modify_LR_patience, mode = 'auto'),
            #     tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = EarlyStop_patience)
            # ],
            batch_size = BATCH_SIZE_LOCAL_FITS,
            verbose = SETTING_DNN_TRAINING_VERBOSE)
        
        training_loss_data = history_of_training.history['loss']
        validation_loss_data = history_of_training.history['val_loss']

        try:
            tensorflow_network.save(f"replica_number_{replica_index + 1}_v{_version_number}.keras")
            print("> Saved replica!" )

        except Exception as error:
            print(f"> Unable to save Replica model replica_number_{replica_index + 1}_v{_version_number}.keras:\n> {error}")
    
        end_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        print(f"> Replica job #{replica_index + 1} finished in {end_time_in_milliseconds - start_time_in_milliseconds}ms.")
        
        # (1): Set up the Figure instance
        figure_instance_nn_loss = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance_nn_loss = figure_instance_nn_loss.add_subplot(1, 1, 1)
        
        plot_customization_nn_loss = PlotCustomizer(
            axis_instance_nn_loss,
            title = "Neural Network Loss",
            xlabel = "Epoch",
            ylabel = "Loss")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = np.array([np.max(training_loss_data) for number in training_loss_data]),
            color = "red",
            linestyle = ':')
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = training_loss_data,
            label = 'Training Loss',
            color = "orange")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = validation_loss_data,
            label = 'Validation Loss',
            color = "pink")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = np.zeros(shape = EPOCHS),
            color = "limegreen",
            linestyle = ':')
        
        figure_instance_nn_loss.savefig(f"loss_replica_{replica_index + 1}_v{_version_number}.png")
        plt.close()