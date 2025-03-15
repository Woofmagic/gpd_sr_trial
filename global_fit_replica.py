# Native Libraries:

# Native Library: | os
import os 

# Native Library | sys
import sys

# Native Library | datetime
import datetime

# Native Library | multiprocessing
import multiprocessing as mp

# Custom Classes:

# Custom Class | utilities > plot_customizer > PlotCustomizer
from utilities.plot_customizer import PlotCustomizer

# Custom Functions:

# Custom Function | networks > global_fit > global_fit_model.py > global_fit_model()
from networks.global_fit.global_fit_model import global_fit_model

# Custom Function | split_data.py > split_data()
from split_data import split_data

# External Libraries:

# External Library | NumPy
import numpy as np

# External Library | Pandas
import pandas as pd

# External Library | TensorFlow
import tensorflow as tf

# External Library | Matplotlib
import matplotlib.pyplot as plt

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

NUMBER_OF_REPLICAS = 15

def run_global_fit_replica_method(
        replica_index: int,
        model_builder: callable,
        data_file: pd.DataFrame,
        version_number: int = 1):
    
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if SETTING_DEBUG:
        print(f"> TensorFlow found GPS: {gpus}.")

    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.set_logical_device_configuration(
    #             gpus[0],
    #             [tf.config.LogicalDeviceConfiguration(memory_limit = 25*1000)])
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # Prevent full memory allocation
            print("> Successfully initialized TensorFlow GPU.")
        except RuntimeError as e:
            print(f"🚨 TensorFlow GPU initialization failed: {e}")
    
    # (1): We used this enough to define it:
    replica_number = replica_index + 1

    # (2): Let the verbose people know what Replica is actually preparing to run:
    if SETTING_VERBOSE:
        print(f"> Now initializing replica #{replica_number}...")

    # (3): Initialize the TF local fitting DNN:
    tensorflow_network = model_builder()

    if SETTING_DEBUG:
        print("> Succesfully loaded TF model!")

    data_file = data_file.copy()

    if SETTING_DEBUG:
        print("> Succesfully copied DataFrame!")

    # (): Generate the pseudodata column for the CFF Re[H]:
    data_file['ReH_pseudo_mean'] = np.random.normal(
        loc = data_file['ReH_pred'],
        scale = data_file['ReH_std'])

    # (): Generate the pseudodata column for the CFF Re[E]:
    data_file['ReE_pseudo_mean'] = np.random.normal(
        loc = data_file['ReE_pred'],
        scale = data_file['ReE_std'])

    # (): Generate the pseudodata column for the CFF Re[Ht]:
    data_file['ReHt_pseudo_mean'] = np.random.normal(
        loc = data_file['ReHt_pred'],
        scale = data_file['ReHt_std'])

    # (): Generate the pseudodata column for the DVCS contribution:
    data_file['dvcs_pseudo_mean'] = np.random.normal(
        loc = data_file['dvcs_pred'],
        scale = data_file['dvcs_std'])

    computed = f'global_fit_replica_{replica_number}_data_v{version_number}.csv'

    # (): Save the file to a .csv for future analysis:
    # - Note that the DataFrame that is saved contains *four* new columns!
    data_file.to_csv(computed, index = False)

    # (): Data Preprocessing | Splitting into {(x_train, y_train)} and {(x_validation, y_validation)}:
    training_x_data, testing_x_data, training_y_data, testing_y_data, training_y_error, testing_y_error = split_data(
        x_data = data_file[['QQ', 'x_b', 't']],
        y_data = data_file[['ReH_pseudo_mean', 'ReE_pseudo_mean', 'ReHt_pseudo_mean', 'dvcs_pseudo_mean']],
        y_error_data = data_file[['ReH_std', 'ReE_std', 'ReHt_std', 'dvcs_std']],
        split_percentage = _TRAIN_VALIDATION_PERCENTAGE)
    
    # (15): Let the debuggers know we split it successfully!
    if SETTING_DEBUG:
        print(f"> Successfully split pseudodata into training and testing data with a train/validation split of {_TRAIN_VALIDATION_PERCENTAGE}")

    figure_cff_real_h_histogram = plt.figure(figsize = (18, 6))
    
    axis_instance_cff_h_histogram = figure_cff_real_h_histogram.add_subplot(1, 1, 1)
    
    plot_customization_cff_pseudodata = PlotCustomizer(
        axis_instance_cff_h_histogram,
        title = r"CFF Pseudodata Sampling Sanity Check for Replica {}".format(replica_number))

    plot_customization_cff_pseudodata.add_errorbar_plot(
        x_data = data_file['set'],
        y_data = data_file['ReH_pred'],
        x_errorbars = np.zeros(data_file['set'].shape),
        y_errorbars = data_file['ReH_std'],
        label = r'Data from Local Fits Re[H]',
        color = "#ff6b63",)

    plot_customization_cff_pseudodata.add_errorbar_plot(
        x_data = data_file['set'] + 0.1,
        y_data = data_file['ReE_pred'],
        x_errorbars = np.zeros(data_file['set'].shape),
        y_errorbars = data_file['ReE_std'],
        label = r'Data from Local Fits Re[E]',
        color = "#ffb663",)

    plot_customization_cff_pseudodata.add_errorbar_plot(
        x_data = data_file['set'] + 0.2,
        y_data = data_file['ReHt_pred'],
        x_errorbars = np.zeros(data_file['set'].shape),
        y_errorbars = data_file['ReHt_std'],
        label = r'Data from Local Fits Re[Ht]',
        color = "#63ff87",)

    plot_customization_cff_pseudodata.add_errorbar_plot(
        x_data = data_file['set'] + 0.3,
        y_data = data_file['dvcs_pred'],
        x_errorbars = np.zeros(data_file['set'].shape),
        y_errorbars = data_file['dvcs_std'],
        label = r'Data from Local Fits DVCS',
        color = "#6392ff",)

    plot_customization_cff_pseudodata.add_errorbar_plot(
        x_data = data_file['set'],
        y_data = data_file['ReH_pseudo_mean'],
        x_errorbars = np.zeros(data_file['ReH_pseudo_mean'].shape),
        y_errorbars = np.zeros(data_file['ReH_pseudo_mean'].shape),
        label = r'Replica Data for Re[H]',
        color = "red",)

    plot_customization_cff_pseudodata.add_errorbar_plot(
        x_data = data_file['set'] + 0.1,
        y_data = data_file['ReE_pseudo_mean'],
        x_errorbars = np.zeros(data_file['ReE_pseudo_mean'].shape),
        y_errorbars = np.zeros(data_file['ReE_pseudo_mean'].shape),
        label = r'Replica Data for Re[E]',
        color = "orange",)

    plot_customization_cff_pseudodata.add_errorbar_plot(
        x_data = data_file['set'] + 0.2,
        y_data = data_file['ReHt_pseudo_mean'],
        x_errorbars = np.zeros(data_file['ReHt_pseudo_mean'].shape),
        y_errorbars = np.zeros(data_file['ReHt_pseudo_mean'].shape),
        label = r'Replica Data for Re[Ht]',
        color = "limegreen",)

    plot_customization_cff_pseudodata.add_errorbar_plot(
        x_data = data_file['set'] + 0.3,
        y_data = data_file['dvcs_pseudo_mean'],
        x_errorbars = np.zeros(data_file['dvcs_pseudo_mean'].shape),
        y_errorbars = np.zeros(data_file['dvcs_pseudo_mean'].shape),
        label = r'Replica Data for DVCS',
        color = "blue",)

    figure_cff_real_h_histogram.savefig(f"global_fit_cff_sampling_replica_{replica_number}_v{version_number}.png")
    plt.close()
    
    start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)

    # (): Let them know we are now running a replica!
    if SETTING_VERBOSE:
        print(f"> Replica #{replica_number} now running...")

    history_of_training = tensorflow_network.fit(
        training_x_data,
        training_y_data,
        validation_data = (testing_x_data, testing_y_data),
        epochs = EPOCHS,
        # callbacks = [
        #     tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = modify_LR_factor, patience = modify_LR_patience, mode = 'auto'),
        #     tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = EarlyStop_patience)
        # ],
        batch_size = BATCH_SIZE_GLOBAL_FITS,
        verbose = SETTING_DNN_TRAINING_VERBOSE)
    
    training_loss_data = history_of_training.history['loss']
    validation_loss_data = history_of_training.history['val_loss']

    replica_keras_model_name = f"global_fit_replica_number_{replica_number}_v{version_number}.keras"

    try:
        tensorflow_network.save(replica_keras_model_name)
        
        if SETTING_DEBUG:
            print(f"> Saved replica under name: {replica_keras_model_name}")

    except Exception as error:
        print(f"> Unable to save Replica model named: {replica_keras_model_name}\n> {error}")

    end_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
    
    if SETTING_VERBOSE:
        print(f"> Replica job #{replica_number} finished in {end_time_in_milliseconds - start_time_in_milliseconds}ms.")

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
    
    computed_figure_dnn_global_fit_loss_name= f"global_fit_loss_replica_{replica_number}_v{version_number}.png"
    figure_instance_nn_loss.savefig(computed_figure_dnn_global_fit_loss_name)
    plt.close()

def parallelized_global_fitting(
        number_of_total_iterations_needed: int,
        version_number: int):
    
    DATA_GLOBAL_FIT_FILE_NAME = "DNN_projections_16_to_30.csv"
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
    figure_instance_cff_h_versus_x_and_t.savefig(f"cff_real_h_vs_xb_and_t_v{version_number}.png")
    figure_instance_cff_e_versus_x_and_t.savefig(f"cff_real_e_vs_xb_and_t_v{version_number}.png")
    figure_instance_cff_ht_versus_x_and_t.savefig(f"cff_real_ht_vs_xb_and_t_v{version_number}.png")
    figure_instance_cff_dvcs_versus_x_and_t.savefig(f"cff_real_dvcs_vs_xb_and_t_v{version_number}.png")
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
    figure_instance_cff_h_versus_x_and_q_squared.savefig(f"cff_real_h_vs_xb_and_q_squared_v{version_number}.png")
    figure_instance_cff_e_versus_x_and_q_squared.savefig(f"cff_real_e_vs_xb_and_q_squared_v{version_number}.png")
    figure_instance_cff_ht_versus_x_and_q_squared.savefig(f"cff_real_ht_vs_xb_and_q_squared_v{version_number}.png")
    figure_instance_cff_dvcs_versus_x_and_q_squared.savefig(f"cff_real_dvcs_vs_xb_and_q_squared_v{version_number}.png")
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
    figure_instance_cff_h_versus_t_and_q_squared.savefig(f"cff_real_h_vs_t_and_q_squared_v{version_number}.png")
    figure_instance_cff_e_versus_t_and_q_squared.savefig(f"cff_real_e_vs_t_and_q_squared_v{version_number}.png")
    figure_instance_cff_ht_versus_t_and_q_squared.savefig(f"cff_real_ht_vs_t_and_q_squared_v{version_number}.png")
    figure_instance_cff_dvcs_versus_t_and_q_squared.savefig(f"cff_real_dvcs_t_and_q_squared_v{version_number}.png")
    plt.close()

    if SETTING_VERBOSE:
        print(f"> Recieved {number_of_total_iterations_needed} iterations needed to complete...")

    number_of_processes = min(
        mp.cpu_count(),
        number_of_total_iterations_needed)
    
    if SETTING_VERBOSE:
        print(f"> mp found {number_of_processes} CPUs to utilize.")

    for replica_index in range(NUMBER_OF_REPLICAS):
        run_global_fit_replica_method(replica_index, global_fit_model, global_fit_data_unique_kinematic_sets, version_number)

    # gpus = tf.config.experimental.list_physical_devices('GPU')

    # if SETTING_DEBUG:
    #     print(f"> TensorFlow found GPS: {gpus}.")

    # for gpu in gpus:

    #     if SETTING_DEBUG:
    #         print(f"> Setting memory growth for GPU: {gpu}.")

    #     tf.config.experimental.set_memory_growth(gpu, True)

    #     if SETTING_VERBOSE:
    #         print(f"> Successfully set memory growth for GPU: {gpu}.")

    # tasks = [
    #     (replica_index, global_fit_model, global_fit_data_unique_kinematic_sets, version_number)
    #     for replica_index in range(NUMBER_OF_REPLICAS)
    # ]

    # if SETTING_DEBUG:
    #     print("> Initialized array of tasks to pass to mp.")

    # with mp.Pool(processes = number_of_processes) as pool:
    #     pool.starmap(run_global_fit_replica_method, tasks, chunksize = 3),

    #     if SETTING_VERBOSE:
    #         print(f"> mp has completed multithread task submission.")

# https://stackoverflow.com/a/18205006
if __name__ == "__main__":
    # parallelized_global_fitting(number_of_total_iterations_needed = 15, version_number = 5)

    version_number = 5

    DATA_GLOBAL_FIT_FILE_NAME = "DNN_projections_16_to_30.csv"
    data_global_fit_for_replica_data = pd.read_csv(DATA_GLOBAL_FIT_FILE_NAME)

    global_fit_data_unique_kinematic_sets = data_global_fit_for_replica_data.groupby('set').first().reset_index()

    try:
        model_paths = [os.path.join(os.getcwd(), file) for file in os.listdir(os.getcwd()) if file.endswith(f"v{version_number}.keras")]
        if SETTING_DEBUG:
            print(f"> Successfully captured {len(model_paths)} in list for iteration.")

    except Exception as error:
        print(f" Error in capturing replicas in list:\n> {error}")
        sys.exit(0)

    if SETTING_VERBOSE:
        print(f"> Obtained {len(model_paths)} models.")


    prediction_inputs = data_global_fit_for_replica_data[['QQ', 'x_b', 't']].to_numpy()

    predictions_for_cffs = []

    for replica_model in model_paths:
        
        if SETTING_VERBOSE:
            print(f"> Now making predictions with replica model: {str(replica_model)}")

        global_model = tf.keras.models.load_model(replica_model)

        predicted_cffs = global_model.predict(prediction_inputs)

        predictions_for_cffs.append(predicted_cffs)

    if SETTING_DEBUG:
        print("> Successfully loaded cross section model!")

    predictions_for_cffs = np.array(predictions_for_cffs)

    # Compute mean and standard deviation over replicas
    mean_predictions = np.mean(predictions_for_cffs, axis=0)  # Shape: (num_samples, 4)
    std_predictions = np.std(predictions_for_cffs, axis=0)    # Shape: (num_samples, 4)

    # Add mean predictions (replica average) to DataFrame
    data_global_fit_for_replica_data[['ReH_pred', 'ReE_pred', 'ReHt_pred', 'dvcs_pred']] = mean_predictions

    # Add standard deviations (replica uncertainty) to DataFrame
    data_global_fit_for_replica_data[['ReH_std', 'ReE_std', 'ReHt_std', 'dvcs_std']] = std_predictions