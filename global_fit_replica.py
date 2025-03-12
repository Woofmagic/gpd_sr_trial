import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from networks.global_fit.global_fit_model import global_fit_model

_version_number = 1

from utilities.plot_customizer import PlotCustomizer

from split_data import split_data

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

NUMBER_OF_REPLICAS = 100


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

DATA_GLOBAL_FIT_FILE_NAME = "global_fit_cumulative_data.csv"
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

def parallel_run():

    num_processes = min(mp.cpu_count(), NUMBER_OF_REPLICAS)

    if SETTING_VERBOSE:
        print(f"> mp found {num_processes} CPUs to utilize.")

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if SETTING_DEBUG:
        print(f"> TensorFlow found GPS: {gpus}.")

    for gpu in gpus:

        if SETTING_DEBUG:
            print(f"> Setting memory growth for GPU: {gpu}.")

        tf.config.experimental.set_memory_growth(gpu, True)

        if SETTING_VERBOSE:
            print(f"> Successfully set memory growth for GPU: {gpu}.")

    tasks = [
        (replica_index, global_fit_model, global_fit_data_unique_kinematic_sets)
        for replica_index in range(NUMBER_OF_REPLICAS)
    ]

    if SETTING_DEBUG:
        print("> Initialized array of tasks to pass to mp.")

    with mp.Pool(processes = num_processes) as pool:
        pool.starmap(run_global_fit_replica_method, tasks)

parallel_run()

# run_global_fit_replica_method(
#     number_of_replicas = 5,
#     model_builder = global_fit_model,
#     data_file = global_fit_data_unique_kinematic_sets)

import os 
import sys

try:
    model_paths = [os.path.join(os.getcwd(), file) for file in os.listdir(os.getcwd()) if file.endswith(f"v{_version_number}.keras")]
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