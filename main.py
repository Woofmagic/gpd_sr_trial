import sys
import os
import re
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from networks.local_fit.local_fit_model import local_fit_model
from networks.global_fit.global_fit_model import global_fit_model

_version_number = 1

from utilities.plot_customizer import PlotCustomizer
            
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

def symbolic_regression(x_data, y_data):
    
    from pysr import PySRRegressor

    _SEARCH_SPACE_BINARY_OPERATORS = [
        "+", "-", "*", "/", "^"
    ]

    _SEARCH_SPACE_UNARY_OPERATORS = [
    "exp", "log", "sqrt", "sin", "cos", "tan"
    ]

    _SEARCH_SPACE_MAXIUMUM_COMPLEXITY = 20

    _SEARCH_SPACE_MAXIMUM_DEPTH = None

    py_regressor_model = PySRRegressor(

    # === SEARCH SPACE ===

    # (1): Binary operators:
    binary_operators = _SEARCH_SPACE_BINARY_OPERATORS,

    # (2): Unary operators:
    unary_operators = _SEARCH_SPACE_UNARY_OPERATORS,

    # (3): Maximum complexity of chosen equation:
    maxsize = _SEARCH_SPACE_MAXIUMUM_COMPLEXITY,

    # (4): Maximum depth of a chosen equation:
    maxdepth = _SEARCH_SPACE_MAXIMUM_DEPTH,

    # === SEARCH SIZE ===

    # (1): Number of iterations for the algorithm:
    niterations = 40,

    # (2): The number of "populations" running:
    populations = 15,

    # (3): The size of each population:
    population_size = 33,

    # (4): Whatever this means:
    ncycles_per_iteration = 550,

    # === OBJECTIVE ===

    # (1): Option 1: Specify *Julia* code to compute elementwise loss:
    elementwise_loss = "loss(prediction, target) = (prediction - target)^2",

    # (2): Option 2: Code your own *Julia* loss:
    loss_function = None,

    # (3): Choose the "metric" to select the final function --- can be 'accuracy,' 'best', or 'score':
    model_selection = 'best',

    # (4): How much to penalize a given function if dim-analysis doesn't work:
    dimensional_constraint_penalty = 1000.0,

    # (5): Enable or disable a search for dimensionless constants:
    dimensionless_constants_only = False,

    # === COMPLEXITY ===

    # (1): Multiplicative factor that penalizes a complex function:
    parsimony = 0.0032,

    # (2): A complicated dictionary governing how complex a given operation can be:
    constraints = None,

    # (3): Another dictionary that enforces the number of times an operator may be nested:
    nested_constraints = None,

    # (4): Another dictionary that limits the complexity per operator:
    complexity_of_operators = None)

    py_regressor_model.fit(x_data, y_data)

    print(py_regressor_model.sympy())

def split_data(x_data, y_data, y_error_data, split_percentage = 0.1):
    """Splits data into training and testing sets based on a random selection of indices."""
    test_indices = np.random.choice(
        x_data.index,
        size = int(len(y_data) * split_percentage),
        replace = False)

    train_X = x_data.loc[~x_data.index.isin(test_indices)]
    test_X = x_data.loc[test_indices]

    train_y = y_data.loc[~y_data.index.isin(test_indices)]
    test_y = y_data.loc[test_indices]

    train_yerr = y_error_data.loc[~y_error_data.index.isin(test_indices)]
    test_yerr = y_error_data.loc[test_indices]

    return train_X, test_X, train_y, test_y, train_yerr, test_yerr

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

def run_local_fit_replica_method(number_of_replicas, model_builder, data_file, kinematic_set_number):

    if SETTING_VERBOSE:
        print(f"> Beginning Replica Method with {number_of_replicas} total Replicas...")

    for replica_index in range(number_of_replicas):

        if SETTING_VERBOSE:
            print(f"> Now initializing replica #{replica_index + 1}...")

        tensorflow_network = model_builder()

        this_replica_data_set = data_file[data_file['set'] == kinematic_set_number].reset_index(drop = True)

        pseudodata_dataframe = generate_replica_data(
            pandas_dataframe = this_replica_data_set,
            mean_value_column_name = 'F',
            stddev_column_name = 'sigmaF',
            new_column_name = 'True_F')

        training_x_data, testing_x_data, training_y_data, testing_y_data, training_y_error, testing_y_error = split_data(
            x_data = pseudodata_dataframe[['QQ', 'x_b', 't', 'phi_x', 'k']],
            y_data = pseudodata_dataframe['F'],
            y_error_data = pseudodata_dataframe['sigmaF'],
            split_percentage = 0.1)
        
        # (1): Set up the Figure instance
        figure_instance_predictions = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance_predictions = figure_instance_predictions.add_subplot(1, 1, 1)
        
        plot_customization_predictions = PlotCustomizer(
            axis_instance_predictions,
            title = r"$\sigma$ vs. $\phi$ for Kinematic Setting {{}}".format(kinematic_set_number),
            xlabel = r"$\phi$",
            ylabel = r"$\sigma$")
        
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
        
        figure_instance_predictions.savefig(f"cross_section_vs_phi_kinematic_set_{kinematic_set_number}_replica_{replica_index}.png")

        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        print(f"> Replica #{replica_index + 1} now running...")

        history_of_training = tensorflow_network.fit(
            training_x_data,
            training_y_data,
            validation_data = (testing_x_data, testing_y_data),
            epochs = EPOCHS,
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = modify_LR_factor, patience = modify_LR_patience, mode = 'auto'),
                tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = EarlyStop_patience)
            ],
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

def run_global_fit_replica_method(number_of_replicas, model_builder, data_file):

    if SETTING_VERBOSE:
        print(f"> Beginning Replica Method with {number_of_replicas} total Replicas...")

    for replica_index in range(number_of_replicas):

        if SETTING_VERBOSE:
            print(f"> Now initializing replica #{replica_index + 1}...")

        tensorflow_network = model_builder()

        if SETTING_DEBUG:
            print("> Succesfully loaded TF model!")

        data_file = data_file.copy()

        # Apply Gaussian sampling for each unique 'set'
        data_file['ReH_pseudo_mean'] = np.random.normal(
            loc = data_file['ReH_pred'],
            scale = data_file['ReH_std'])

        data_file['ReE_pseudo_mean'] = np.random.normal(
            loc = data_file['ReE_pred'],
            scale = data_file['ReE_std'])

        data_file['ReHt_pseudo_mean'] = np.random.normal(
            loc = data_file['ReHt_pred'],
            scale = data_file['ReHt_std'])

        data_file['dvcs_pseudo_mean'] = np.random.normal(
            loc = data_file['dvcs_pred'],
            scale = data_file['dvcs_std'])

        data_file.to_csv(f'global_fit_replica_{replica_index + 1}_data_v{_version_number}.csv', index = False)

        training_x_data, testing_x_data, training_y_data, testing_y_data, training_y_error, testing_y_error = split_data(
            x_data = data_file[['QQ', 'x_b', 't']],
            y_data = data_file[['ReH_pseudo_mean', 'ReE_pseudo_mean', 'ReHt_pseudo_mean', 'dvcs_pseudo_mean']],
            y_error_data = data_file[['ReH_std', 'ReE_std', 'ReHt_std', 'dvcs_std']],
            split_percentage = 0.1)
        
        figure_cff_real_h_histogram = plt.figure(figsize = (18, 6))
        axis_instance_cff_h_histogram = figure_cff_real_h_histogram.add_subplot(1, 1, 1)
        plot_customization_cff_h_histogram = PlotCustomizer(
            axis_instance_cff_h_histogram,
            title = r"CFF Pseudodata Sampling Sanity Check")

        plot_customization_cff_h_histogram.add_errorbar_plot(
            x_data = data_file['set'],
            y_data = data_file['ReH_pred'],
            x_errorbars = np.zeros(data_file['set'].shape),
            y_errorbars = data_file['ReH_std'],
            label = r'Data from Local Fits Re[H]',
            color = "#ff6b63",)

        plot_customization_cff_h_histogram.add_errorbar_plot(
            x_data = data_file['set'] + 0.1,
            y_data = data_file['ReE_pred'],
            x_errorbars = np.zeros(data_file['set'].shape),
            y_errorbars = data_file['ReE_std'],
            label = r'Data from Local Fits Re[E]',
            color = "#ffb663",)

        plot_customization_cff_h_histogram.add_errorbar_plot(
            x_data = data_file['set'] + 0.2,
            y_data = data_file['ReHt_pred'],
            x_errorbars = np.zeros(data_file['set'].shape),
            y_errorbars = data_file['ReHt_std'],
            label = r'Data from Local Fits Re[Ht]',
            color = "#63ff87",)

        plot_customization_cff_h_histogram.add_errorbar_plot(
            x_data = data_file['set'] + 0.3,
            y_data = data_file['dvcs_pred'],
            x_errorbars = np.zeros(data_file['set'].shape),
            y_errorbars = data_file['dvcs_std'],
            label = r'Data from Local Fits DVCS',
            color = "#6392ff",)

        plot_customization_cff_h_histogram.add_errorbar_plot(
            x_data = data_file['set'],
            y_data = data_file['ReH_pseudo_mean'],
            x_errorbars = np.zeros(data_file['ReH_pseudo_mean'].shape),
            y_errorbars = np.zeros(data_file['ReH_pseudo_mean'].shape),
            label = r'Replica Data for Re[H]',
            color = "red",)

        plot_customization_cff_h_histogram.add_errorbar_plot(
            x_data = data_file['set'] + 0.1,
            y_data = data_file['ReE_pseudo_mean'],
            x_errorbars = np.zeros(data_file['ReE_pseudo_mean'].shape),
            y_errorbars = np.zeros(data_file['ReE_pseudo_mean'].shape),
            label = r'Replica Data for Re[E]',
            color = "orange",)

        plot_customization_cff_h_histogram.add_errorbar_plot(
            x_data = data_file['set'] + 0.2,
            y_data = data_file['ReHt_pseudo_mean'],
            x_errorbars = np.zeros(data_file['ReHt_pseudo_mean'].shape),
            y_errorbars = np.zeros(data_file['ReHt_pseudo_mean'].shape),
            label = r'Replica Data for Re[Ht]',
            color = "limegreen",)

        plot_customization_cff_h_histogram.add_errorbar_plot(
            x_data = data_file['set'] + 0.3,
            y_data = data_file['dvcs_pseudo_mean'],
            x_errorbars = np.zeros(data_file['dvcs_pseudo_mean'].shape),
            y_errorbars = np.zeros(data_file['dvcs_pseudo_mean'].shape),
            label = r'Replica Data for DVCS',
            color = "blue",)

        figure_cff_real_h_histogram.savefig(f"global_fit_cff_sampling_replica_{replica_index + 1}_v{_version_number}.png")
        plt.close()
        
        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        print(f"> Replica #{replica_index + 1} now running...")

        history_of_training = tensorflow_network.fit(
            training_x_data,
            training_y_data,
            validation_data = (testing_x_data, testing_y_data),
            epochs = EPOCHS,
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = modify_LR_factor, patience = modify_LR_patience, mode = 'auto'),
                tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = EarlyStop_patience)
            ],
            batch_size = BATCH_SIZE_GLOBAL_FITS,
            verbose = SETTING_DNN_TRAINING_VERBOSE)
        
        training_loss_data = history_of_training.history['loss']
        validation_loss_data = history_of_training.history['val_loss']

        try:
            tensorflow_network.save(f"global_fit_replica_number_{replica_index + 1}_v{_version_number}.keras")
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
        
        figure_instance_nn_loss.savefig(f"global_fit_loss_replica_{replica_index + 1}_v{_version_number}.png")
        plt.close()

def density_scatter(x, y, ax, bins = 50, cmap = 'viridis'):
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xidx = np.digitize(x, xedges[:-1]) - 1
    yidx = np.digitize(y, yedges[:-1]) - 1
    density = counts[xidx, yidx]
    scatter = ax.scatter(x, y, c = density, cmap = cmap, s = 10, alpha = 1.0, edgecolor = 'none')
    return scatter

def get_next_version(base_path: str) -> str:
    """
    Scans the base_path (e.g., 'science/analysis/' or 'science/data/') to find the highest version_x directory,
    then returns the next version path.
    """

    if not os.path.exists(base_path):
        os.makedirs(base_path)  # Ensure the base path exists
    
    # Regex to match 'version_x' pattern
    version_pattern = re.compile(r'version_(\d+)')
    
    existing_versions = []
    for entry in os.listdir(base_path):
        match = version_pattern.match(entry)
        if match:
            existing_versions.append(int(match.group(1)))
    
    next_version = max(existing_versions, default=-1) + 1
    return next_version

def run():
    
    _PATH_SCIENCE_ANALYSIS = 'science/analysis/'
    _PATH_SCIENCE_DATA = 'science/data'

    # Get next version directories
    _version_number = get_next_version(_PATH_SCIENCE_ANALYSIS)

    print(f"> Determined next analysis directory: {_version_number}")

    try:
        # tf.config.set_visible_devices([],'GPU')
        tensorflow_found_devices = tf.config.list_physical_devices()

        if len(tf.config.list_physical_devices()) != 0:
            for device in tensorflow_found_devices:
                print(f"> TensorFlow detected device: {device}")

        else:
            print("> TensorFlow didn't find CPUs or GPUs...")

    except Exception as error:
        print(f"> TensorFlow could not find devices due to error:\n> {error}")
    
    DATA_FILE_NAME = "data.csv"
    data_file = pd.read_csv(DATA_FILE_NAME)

    if SETTING_DEBUG:
        print(f"> Obtained data file with experimental data called: {DATA_FILE_NAME}")

    data_file = data_file[data_file["F"] != 0]
    kinematic_sets = sorted(data_file["set"].unique())

    # First, we analyze the kinematic phase space in the data:
    figure_instance_kinematic_xb_versus_q_squared = plt.figure(figsize = (8, 6))
    axis_instance_fitting_xb_versus_q_squared = figure_instance_kinematic_xb_versus_q_squared.add_subplot(1, 1, 1)
    plot_customization_data_comparison = PlotCustomizer(
        axis_instance_fitting_xb_versus_q_squared,
        title = r"[Experimental] Kinematic Phase Space in $x_{{B}}$ and $Q^{{2}}$",
        xlabel = r"$x_{{B}}$",
        ylabel = r"Q^{{2}}",
        xlim = (0.0, 0.6),
        ylim = (0.5, 5.0),
        grid = True)
    density_scatter(data_file['x_b'], data_file['QQ'], axis_instance_fitting_xb_versus_q_squared)
    figure_instance_kinematic_xb_versus_q_squared.savefig(f"phase_space_in_xb_QQ_v{_version_number}.png")
    plt.close()

    figure_instance_kinematic_xb_versus_t = plt.figure(figsize = (8, 6))
    axis_instance_fitting_xb_versus_t = figure_instance_kinematic_xb_versus_t.add_subplot(1, 1, 1)
    plot_customization_data_comparison = PlotCustomizer(
        axis_instance_fitting_xb_versus_q_squared,
        title = r"[Experimental] Kinematic Phase Space in $x_{{B}}$ and $-t$",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$-t$",
        xlim = (0.0, 0.6),
        grid = True)
    density_scatter(data_file['x_b'], -data_file['t'], axis_instance_fitting_xb_versus_t)
    figure_instance_kinematic_xb_versus_q_squared.savefig(f"phase_space_in_xb_t_v{_version_number}.png")
    plt.close()

    figure_instance_kinematic_phase_space = plt.figure(figsize = (8, 6))
    axis_instance_fitting_phase_space = figure_instance_kinematic_phase_space.add_subplot(1, 1, 1, projection = '3d')
    plot_customization_data_comparison = PlotCustomizer(
        axis_instance_fitting_phase_space,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$-t$",
        grid = True)
    plot_customization_data_comparison.add_3d_scatter_plot(
        x_data = data_file['x_b'],
        y_data = data_file['QQ'],
        z_data = -data_file['t'],
        color = 'royalblue',
        marker = '.')
    figure_instance_kinematic_phase_space.savefig(f"phase_space_v{_version_number}.png")
    plt.close()
    
    # We now perform local fits:
    # for kinematic_set_number in kinematic_sets:
    RESTRICTED_KINEMATIC_SETS_FOR_TESTING = [1.0, 2.0, 3.0, 4.0]
    for kinematic_set_number in RESTRICTED_KINEMATIC_SETS_FOR_TESTING:

        if SETTING_VERBOSE:
            print(f"> Now running kinematic set number {kinematic_set_number}...")

        run_local_fit_replica_method(
            number_of_replicas = 1,
            model_builder = local_fit_model,
            data_file = data_file,
            kinematic_set_number = kinematic_set_number)

        if SETTING_VERBOSE:
            print(f"> Finished running Replica Method on kinematic set number {kinematic_set_number}!")

    available_kinematic_sets = [1, 2, 3, 4]
    for available_kinematic_set in available_kinematic_sets:

        if SETTING_VERBOSE:
            print(f"> Now generating .csv files for kinematic set #{available_kinematic_set}...")

        try:
            model_paths = [os.path.join(os.getcwd(), file) for file in os.listdir(os.getcwd()) if file.endswith(f"v{_version_number}.keras")]
            if SETTING_DEBUG:
                print(f"> Successfully  captured {len(model_paths)} in list for iteration.")

        except Exception as error:
            print(f" Error in capturing replicas in list:\n> {error}")
            sys.exit(0)

        if SETTING_VERBOSE:
            print(f"> Obtained {len(model_paths)} models.")

        dataframe_restricted_to_current_kinematic_set = data_file[data_file['set'] == available_kinematic_set]
        dataframe_restricted_to_current_kinematic_set = dataframe_restricted_to_current_kinematic_set

        prediction_inputs = dataframe_restricted_to_current_kinematic_set[['QQ', 'x_b', 't', 'phi_x', 'k']].to_numpy()
        real_cross_section_values = dataframe_restricted_to_current_kinematic_set['F'].values
        phi_values = dataframe_restricted_to_current_kinematic_set['phi_x'].values
    
        # These contain *per replica* predictions
        predictions_for_cross_section = []
        predictions_for_cffs = []

        predictions_for_cross_section_average = []
        predictions_for_cffs_average = []
        predictions_for_cross_section_std_dev = []

        for replica_models in model_paths:

            if SETTING_VERBOSE:
                print(f"> Now making predictions with replica model: {str(replica_models)}")
    
            # This just loads a TF model file:
            cross_section_model = tf.keras.models.load_model(replica_models, custom_objects = {'TotalFLayer': TotalFLayer})
            
            if SETTING_DEBUG:
                print("> Successfully loaded cross section model!")

            # This defines a *new* TF model:
            cff_model = tf.keras.Model(
                inputs = cross_section_model.input,
                outputs = cross_section_model.get_layer('cff_output_layer').output)
            
            if SETTING_DEBUG:
                print("> Successfully CFF submodel!")

            predicted_cffs = cff_model.predict(prediction_inputs)
            predicted_cross_sections = cross_section_model.predict(prediction_inputs)

            predictions_for_cross_section.append(predicted_cffs)
            predictions_for_cffs.append(predicted_cross_sections)

        for azimuthal_angle in range(len(phi_values)):

            if SETTING_DEBUG:
                print(f"> Now analyzing averages at azimuthal angle of {azimuthal_angle} degrees...")

            cross_section_at_given_phi = [sigma[azimuthal_angle] for sigma in predictions_for_cross_section]
            predictions_for_cross_section_average.append(np.mean(cross_section_at_given_phi))
            predictions_for_cross_section_std_dev.append(np.std(cross_section_at_given_phi))
            
        predictions_for_cross_section_average = np.array(predictions_for_cross_section_average)
        predictions_for_cross_section_std_dev = np.array(predictions_for_cross_section_std_dev)

        chi_squared_error = np.sum(((real_cross_section_values - predictions_for_cross_section_average) / predictions_for_cross_section_std_dev) ** 2)
        chi_square_file = "chi2.txt"
        if not os.path.exists(chi_square_file):
            with open(chi_square_file, 'w') as file:
                file.write("Kinematic Set\tChi-Square Error\n")  # Write header if the file doesn't exist
        # Append chi-square error data to the file
        with open(chi_square_file, 'a') as file:
            file.write(f"{available_kinematic_set}\t{chi_squared_error:.4f}\n")
        print(f"Kinematic Set {available_kinematic_set}: Chi-Square Error = {chi_squared_error:.4f}")

        f_vs_phi_data = {
            'azimuthal_phi': phi_values,
            'cross_section': real_cross_section_values,
            'cross_section_average_prediction': predictions_for_cross_section_average,
            'cross_section_std_dev_prediction': predictions_for_cross_section_std_dev
        }

        f_vs_phi_df = pd.DataFrame(f_vs_phi_data)
        f_vs_phi_df.to_csv('fuczzzk.csv', index=False)

        # (1): Set up the Figure instance
        figure_cff_real_h_histogram = plt.figure(figsize = (18, 6))
        figure_cff_real_e_histogram = plt.figure(figsize = (18, 6))
        figure_cff_real_ht_histogram = plt.figure(figsize = (18, 6))
        figure_cff_dvcs_histogram = plt.figure(figsize = (18, 6))
        axis_instance_cff_h_histogram = figure_cff_real_h_histogram.add_subplot(1, 1, 1)
        axis_instance_cff_e_histogram = figure_cff_real_e_histogram.add_subplot(1, 1, 1)
        axis_instance_cff_ht_histogram = figure_cff_real_ht_histogram.add_subplot(1, 1, 1)
        axis_instance_cff_dvcs_histogram = figure_cff_dvcs_histogram.add_subplot(1, 1, 1)
        plot_customization_cff_h_histogram = PlotCustomizer(
            axis_instance_cff_h_histogram,
            title = r"Predictions for $Re \left(H \right)$")
        plot_customization_cff_e_histogram = PlotCustomizer(
            axis_instance_cff_e_histogram,
            title = r"Predictions for $Re \left(E \right)$")
        plot_customization_cff_ht_histogram = PlotCustomizer(
            axis_instance_cff_ht_histogram,
            title = r"Predictions for $Re \left(\tilde{H} \right)$")
        plot_customization_cff_dvcs_histogram = PlotCustomizer(
            axis_instance_cff_dvcs_histogram,
            title = r"Predictions for $DVCS$")
        plot_customization_cff_h_histogram.add_bar_plot(
            x_data = np.array(predictions_for_cffs)[:, :, 1].T.flatten(),
            bins = 20,
            label = "Histogram Bars",
            color = "lightblue",
            use_histogram = True)
        plot_customization_cff_e_histogram.add_bar_plot(
            x_data = np.array(predictions_for_cffs)[:, :, 2].T.flatten(),
            bins = 20,
            label = "Histogram Bars",
            color = "lightblue",
            use_histogram = True)
        plot_customization_cff_ht_histogram.add_bar_plot(
            x_data = np.array(predictions_for_cffs)[:, :, 3].T.flatten(),
            bins = 20,
            label = "Histogram Bars",
            color = "lightblue",
            use_histogram = True)
        plot_customization_cff_dvcs_histogram.add_bar_plot(
            x_data = np.array(predictions_for_cffs)[:, :, 4].T.flatten(),
            bins = 20,
            label = "Histogram Bars",
            color = "lightblue",
            use_histogram = True)
        figure_cff_real_h_histogram.savefig(f"local_fit_cff_real_h_{_version_number}.png")
        figure_cff_real_e_histogram.savefig(f"local_fit_cff_real_e_{_version_number}.png")
        figure_cff_real_ht_histogram.savefig(f"local_fit_cff_real_ht_{_version_number}.png")
        figure_cff_dvcs_histogram.savefig(f"local_fit_cff_dvcs_{_version_number}.png")
        plt.close()

    #### GLOBAL FIT ####

    DATA_GLOBAL_FIT_FILE_NAME = "DNN_projections_1_to_10_no_duplicates.csv"
    data_global_fit_for_replica_data = pd.read_csv(DATA_GLOBAL_FIT_FILE_NAME)

    global_fit_data_unique_kinematic_sets = data_global_fit_for_replica_data.groupby('set').first().reset_index()

    figure_instance_cff_h_versus_x_and_t = plt.figure(figsize = (8, 6))
    figure_instance_cff_e_versus_x_and_t = plt.figure(figsize = (8, 6))
    figure_instance_cff_ht_versus_x_and_t = plt.figure(figsize = (8, 6))
    figure_instance_cff_dvcs_versus_x_and_t = plt.figure(figsize = (8, 6))
    axis_instance_cff_h_versus_x_and_t = figure_instance_cff_h_versus_x_and_t.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_e_versus_x_and_t = figure_instance_cff_e_versus_x_and_t.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_ht_versus_x_and_t = figure_instance_cff_ht_versus_x_and_t.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_dvcs_versus_x_and_t = figure_instance_cff_dvcs_versus_x_and_t.add_subplot(1, 1, 1, projection = '3d')
    plot_customization_cff_h_versus_x_and_t = PlotCustomizer(
        axis_instance_cff_h_versus_x_and_t,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$-t$",
        zlabel = r"$Re[H]$",
        grid = True)
    plot_customization_cff_e_versus_x_and_t = PlotCustomizer(
        axis_instance_cff_e_versus_x_and_t,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$-t$",
        zlabel = r"$Re[E]$",
        grid = True)
    plot_customization_cff_ht_versus_x_and_t = PlotCustomizer(
        axis_instance_cff_ht_versus_x_and_t,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$-t$",
        zlabel = r"$Re[Ht]$",
        grid = True)
    plot_customization_cff_dvcs_versus_x_and_t = PlotCustomizer(
        axis_instance_cff_dvcs_versus_x_and_t,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$-t$",
        zlabel = r"$DVCS$",
        grid = True)
    plot_customization_cff_h_versus_x_and_t.add_3d_scatter_plot(
        x_data = global_fit_data_unique_kinematic_sets['x_b'],
        y_data = -global_fit_data_unique_kinematic_sets['t'],
        z_data = global_fit_data_unique_kinematic_sets['ReH_pred'],
        color = 'red',
        marker = '.')
    for xB, t, cff_value in zip(global_fit_data_unique_kinematic_sets['x_b'], -global_fit_data_unique_kinematic_sets['t'], global_fit_data_unique_kinematic_sets['ReH_pred']):
        plot_customization_cff_h_versus_x_and_t.add_line_plot([xB, xB], [t, t], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_e_versus_x_and_t.add_3d_scatter_plot(
        x_data = global_fit_data_unique_kinematic_sets['x_b'],
        y_data = -global_fit_data_unique_kinematic_sets['t'],
        z_data = global_fit_data_unique_kinematic_sets['ReE_pred'],
        color = 'red',
        marker = '.')
    for xB, t, cff_value in zip(global_fit_data_unique_kinematic_sets['x_b'], -global_fit_data_unique_kinematic_sets['t'], global_fit_data_unique_kinematic_sets['ReE_pred']):
        plot_customization_cff_e_versus_x_and_t.add_line_plot([xB, xB], [t, t], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_ht_versus_x_and_t.add_3d_scatter_plot(
        x_data = global_fit_data_unique_kinematic_sets['x_b'],
        y_data = -global_fit_data_unique_kinematic_sets['t'],
        z_data = global_fit_data_unique_kinematic_sets['ReHt_pred'],
        color = 'red',
        marker = '.')
    for xB, t, cff_value in zip(global_fit_data_unique_kinematic_sets['x_b'], -global_fit_data_unique_kinematic_sets['t'], global_fit_data_unique_kinematic_sets['ReHt_pred']):
        plot_customization_cff_ht_versus_x_and_t.add_line_plot([xB, xB], [t, t], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_dvcs_versus_x_and_t.add_3d_scatter_plot(
        x_data = global_fit_data_unique_kinematic_sets['x_b'],
        y_data = -global_fit_data_unique_kinematic_sets['t'],
        z_data = global_fit_data_unique_kinematic_sets['dvcs_pred'],
        color = 'red',
        marker = '.')
    for xB, t, cff_value in zip(global_fit_data_unique_kinematic_sets['x_b'], -global_fit_data_unique_kinematic_sets['t'], global_fit_data_unique_kinematic_sets['dvcs_pred']):
        plot_customization_cff_dvcs_versus_x_and_t.add_line_plot([xB, xB], [t, t], [0, cff_value], color='#fa9696', linestyle='dashed')
    figure_instance_cff_h_versus_x_and_t.savefig(f"cff_real_h_vs_xb_and_t_v{_version_number}.png")
    figure_instance_cff_e_versus_x_and_t.savefig(f"cff_real_e_vs_xb_and_t_v{_version_number}.png")
    figure_instance_cff_ht_versus_x_and_t.savefig(f"cff_real_ht_vs_xb_and_t_v{_version_number}.png")
    figure_instance_cff_dvcs_versus_x_and_t.savefig(f"cff_real_dvcs_vs_xb_and_t_v{_version_number}.png")
    plt.close()

    figure_instance_cff_h_versus_x_and_q_squared = plt.figure(figsize = (8, 6))
    figure_instance_cff_e_versus_x_and_q_squared = plt.figure(figsize = (8, 6))
    figure_instance_cff_ht_versus_x_and_q_squared = plt.figure(figsize = (8, 6))
    figure_instance_cff_dvcs_versus_x_and_q_squared = plt.figure(figsize = (8, 6))
    axis_instance_cff_h_versus_x_and_q_squared = figure_instance_cff_h_versus_x_and_q_squared.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_e_versus_x_and_q_squared = figure_instance_cff_e_versus_x_and_q_squared.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_ht_versus_x_and_q_squared = figure_instance_cff_ht_versus_x_and_q_squared.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_dvcs_versus_x_and_q_squared = figure_instance_cff_dvcs_versus_x_and_q_squared.add_subplot(1, 1, 1, projection = '3d')
    plot_customization_cff_h_versus_x_and_q_squared = PlotCustomizer(
        axis_instance_cff_h_versus_x_and_q_squared,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$Re[H]$",
        grid = True)
    plot_customization_cff_e_versus_x_and_q_squared = PlotCustomizer(
        axis_instance_cff_e_versus_x_and_q_squared,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$Re[E]$",
        grid = True)
    plot_customization_cff_ht_versus_x_and_q_squared = PlotCustomizer(
        axis_instance_cff_ht_versus_x_and_q_squared,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$Re[Ht]$",
        grid = True)
    plot_customization_cff_dvcs_versus_x_and_q_squared = PlotCustomizer(
        axis_instance_cff_dvcs_versus_x_and_q_squared,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$DVCS$",
        grid = True)
    plot_customization_cff_h_versus_x_and_q_squared.add_3d_scatter_plot(
        x_data = global_fit_data_unique_kinematic_sets['x_b'],
        y_data = global_fit_data_unique_kinematic_sets['QQ'],
        z_data = global_fit_data_unique_kinematic_sets['ReH_pred'],
        color = 'red',
        marker = '.')
    for xB, Q, cff_value in zip(global_fit_data_unique_kinematic_sets['x_b'], global_fit_data_unique_kinematic_sets['QQ'], global_fit_data_unique_kinematic_sets['ReH_pred']):
        plot_customization_cff_h_versus_x_and_q_squared.add_line_plot([xB, xB], [Q, Q], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_e_versus_x_and_q_squared.add_3d_scatter_plot(
        x_data = global_fit_data_unique_kinematic_sets['x_b'],
        y_data = global_fit_data_unique_kinematic_sets['QQ'],
        z_data = global_fit_data_unique_kinematic_sets['ReE_pred'],
        color = 'red',
        marker = '.')
    for xB, Q, cff_value in zip(global_fit_data_unique_kinematic_sets['x_b'], global_fit_data_unique_kinematic_sets['QQ'], global_fit_data_unique_kinematic_sets['ReE_pred']):
        plot_customization_cff_e_versus_x_and_q_squared.add_line_plot([xB, xB], [Q, Q], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_ht_versus_x_and_q_squared.add_3d_scatter_plot(
        x_data = global_fit_data_unique_kinematic_sets['x_b'],
        y_data = global_fit_data_unique_kinematic_sets['QQ'],
        z_data = global_fit_data_unique_kinematic_sets['ReHt_pred'],
        color = 'red',
        marker = '.')
    for xB, Q, cff_value in zip(global_fit_data_unique_kinematic_sets['x_b'], global_fit_data_unique_kinematic_sets['QQ'], global_fit_data_unique_kinematic_sets['ReHt_pred']):
        plot_customization_cff_ht_versus_x_and_q_squared.add_line_plot([xB, xB], [Q, Q], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_dvcs_versus_x_and_q_squared.add_3d_scatter_plot(
        x_data = global_fit_data_unique_kinematic_sets['x_b'],
        y_data = global_fit_data_unique_kinematic_sets['QQ'],
        z_data = global_fit_data_unique_kinematic_sets['dvcs_pred'],
        color = 'red',
        marker = '.')
    for xB, Q, cff_value in zip(global_fit_data_unique_kinematic_sets['x_b'], global_fit_data_unique_kinematic_sets['QQ'], global_fit_data_unique_kinematic_sets['dvcs_pred']):
        plot_customization_cff_dvcs_versus_x_and_q_squared.add_line_plot([xB, xB], [Q, Q], [0, cff_value], color='#fa9696', linestyle='dashed')
    figure_instance_cff_h_versus_x_and_q_squared.savefig(f"cff_real_h_vs_xb_and_q_squared_v{_version_number}.png")
    figure_instance_cff_e_versus_x_and_q_squared.savefig(f"cff_real_e_vs_xb_and_q_squared_v{_version_number}.png")
    figure_instance_cff_ht_versus_x_and_q_squared.savefig(f"cff_real_ht_vs_xb_and_q_squared_v{_version_number}.png")
    figure_instance_cff_dvcs_versus_x_and_q_squared.savefig(f"cff_real_dvcs_vs_xb_and_q_squared_v{_version_number}.png")
    plt.close()

    figure_instance_cff_h_versus_t_and_q_squared = plt.figure(figsize = (8, 6))
    figure_instance_cff_e_versus_t_and_q_squared = plt.figure(figsize = (8, 6))
    figure_instance_cff_ht_versus_t_and_q_squared = plt.figure(figsize = (8, 6))
    figure_instance_cff_dvcs_versus_t_and_q_squared = plt.figure(figsize = (8, 6))
    axis_instance_cff_h_versus_t_and_q_squared = figure_instance_cff_h_versus_t_and_q_squared.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_e_versus_t_and_q_squared = figure_instance_cff_e_versus_t_and_q_squared.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_ht_versus_t_and_q_squared = figure_instance_cff_ht_versus_t_and_q_squared.add_subplot(1, 1, 1, projection = '3d')
    axis_instance_cff_dvcs_versus_t_and_q_squared = figure_instance_cff_dvcs_versus_t_and_q_squared.add_subplot(1, 1, 1, projection = '3d')
    plot_customization_cff_h_versus_t_and_q_squared = PlotCustomizer(
        axis_instance_cff_h_versus_t_and_q_squared,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$Re[H]$",
        grid = True)
    plot_customization_cff_e_versus_t_and_q_squared = PlotCustomizer(
        axis_instance_cff_e_versus_t_and_q_squared,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$Re[E]$",
        grid = True)
    plot_customization_cff_ht_versus_t_and_q_squared = PlotCustomizer(
        axis_instance_cff_ht_versus_t_and_q_squared,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$Re[Ht]$",
        grid = True)
    plot_customization_cff_dvcs_versus_t_and_q_squared = PlotCustomizer(
        axis_instance_cff_dvcs_versus_t_and_q_squared,
        title = r"[Experimental] Kinematic Phase Space",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$DVCS$",
        grid = True)
    plot_customization_cff_h_versus_t_and_q_squared.add_3d_scatter_plot(
        x_data = -global_fit_data_unique_kinematic_sets['t'],
        y_data = global_fit_data_unique_kinematic_sets['QQ'],
        z_data = global_fit_data_unique_kinematic_sets['ReH_pred'],
        color = 'red',
        marker = '.')
    for t, Q, cff_value in zip(-global_fit_data_unique_kinematic_sets['t'], global_fit_data_unique_kinematic_sets['QQ'], global_fit_data_unique_kinematic_sets['ReH_pred']):
        plot_customization_cff_h_versus_t_and_q_squared.add_line_plot([t, t], [Q, Q], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_e_versus_t_and_q_squared.add_3d_scatter_plot(
        x_data = -global_fit_data_unique_kinematic_sets['t'],
        y_data = global_fit_data_unique_kinematic_sets['QQ'],
        z_data = global_fit_data_unique_kinematic_sets['ReE_pred'],
        color = 'red',
        marker = '.')
    for t, Q, cff_value in zip(-global_fit_data_unique_kinematic_sets['t'], global_fit_data_unique_kinematic_sets['QQ'], global_fit_data_unique_kinematic_sets['ReE_pred']):
        plot_customization_cff_e_versus_t_and_q_squared.add_line_plot([t, t], [Q, Q], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_ht_versus_t_and_q_squared.add_3d_scatter_plot(
        x_data = -global_fit_data_unique_kinematic_sets['t'],
        y_data = global_fit_data_unique_kinematic_sets['QQ'],
        z_data = global_fit_data_unique_kinematic_sets['ReHt_pred'],
        color = 'red',
        marker = '.')
    for t, Q, cff_value in zip(-global_fit_data_unique_kinematic_sets['t'], global_fit_data_unique_kinematic_sets['QQ'], global_fit_data_unique_kinematic_sets['ReHt_pred']):
        plot_customization_cff_ht_versus_t_and_q_squared.add_line_plot([t, t], [Q, Q], [0, cff_value], color='#fa9696', linestyle='dashed')
    plot_customization_cff_dvcs_versus_t_and_q_squared.add_3d_scatter_plot(
        x_data = -global_fit_data_unique_kinematic_sets['t'],
        y_data = global_fit_data_unique_kinematic_sets['QQ'],
        z_data = global_fit_data_unique_kinematic_sets['dvcs_pred'],
        color = 'red',
        marker = '.')
    for t, Q, cff_value in zip(-global_fit_data_unique_kinematic_sets['t'], global_fit_data_unique_kinematic_sets['QQ'], global_fit_data_unique_kinematic_sets['dvcs_pred']):
        plot_customization_cff_dvcs_versus_t_and_q_squared.add_line_plot([t, t], [Q, Q], [0, cff_value], color='#fa9696', linestyle='dashed')
    figure_instance_cff_h_versus_t_and_q_squared.savefig(f"cff_real_h_vs_t_and_q_squared_v{_version_number}.png")
    figure_instance_cff_e_versus_t_and_q_squared.savefig(f"cff_real_e_vs_t_and_q_squared_v{_version_number}.png")
    figure_instance_cff_ht_versus_t_and_q_squared.savefig(f"cff_real_ht_vs_t_and_q_squared_v{_version_number}.png")
    figure_instance_cff_dvcs_versus_t_and_q_squared.savefig(f"cff_real_dvcs_t_and_q_squared_v{_version_number}.png")
    plt.close()

    run_global_fit_replica_method(
        number_of_replicas = 300,
        model_builder = global_fit_model,
        data_file = global_fit_data_unique_kinematic_sets)

    #### SR ####
    
    import pysr
    from pysr import PySRRegressor

    # cross_section_model = tf.keras.models.load_model(f"replica_number_1_v{_version_number}.keras")
    # cff_tf_model = tf.keras.Model(
    #             inputs = cross_section_model.input,
    #             outputs = cross_section_model.get_layer('cff_output_layer').output)
            
    # if SETTING_DEBUG:
    #     print("> Successfully retrieved CFF submodel!")

    # X_train = np.column_stack([
    #     data_file['k'],
    #     data_file['QQ'],
    #     data_file['x_b'],
    #     data_file['t'],
    #     data_file['phi_x']
    #     ])
    # Y_train_cffs = cff_tf_model.predict(prediction_inputs)
    # Y_train_cross_section = cross_section_model.predict(prediction_inputs)   

    # cff_model = PySRRegressor(
    # niterations=1000,  # Adjust based on complexity
    # binary_operators=["+", "-", "*", "/"],  # Allowed operations
    # unary_operators=["exp", "log", "sin", "cos"],  # Allowed functions
    # extra_sympy_mappings={},  # Custom functions if needed
    # model_selection="best",  # Choose the simplest best-performing model
    # progress=True,)

    # cff_model.fit(X_train, Y_train_cffs)  # Fit symbolic regression model

    # X_train_extended = np.hstack([X_train, Y_train_cffs])  # Append CFFs as additional inputs

    # global_fit_dnn_model = tf.keras.models.load_model(f"global_fit_replica_number_1_v{_version_number}.keras")

    # cross_section_model = PySRRegressor(
    #     niterations=1000,
    #     binary_operators=["+", "-", "*", "/"],
    #     unary_operators=["exp", "log", "sin", "cos"],
    #     model_selection="best",
    #     progress=True)
    

if __name__ == "__main__":

    run()