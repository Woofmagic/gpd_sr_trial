import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, rc_context
plt.rcParams.update(plt.rcParamsDefault)

# def GlobalFitDNNmodel():
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

#     # Input shape for multiple kinematic values
#     inputs = tf.keras.Input(shape=(5), name='input_layer')  # (QQ, x_b, t, phi, k)
#     QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)

#     # Concatenate QQ, x_b, t for processing
#     kinematics = tf.keras.layers.concatenate([QQ, x_b, t])

#     # Neural network layers to predict CFFs
#     x1 = tf.keras.layers.Dense(480, activation="relu6", kernel_initializer=initializer)(kinematics)
#     x2 = tf.keras.layers.Dense(320, activation="relu6", kernel_initializer=initializer)(x1)
#     x3 = tf.keras.layers.Dense(240, activation="relu6", kernel_initializer=initializer)(x2)
#     x4 = tf.keras.layers.Dense(120, activation="relu6", kernel_initializer=initializer)(x3)
#     x5 = tf.keras.layers.Dense(32, activation="relu6", kernel_initializer=initializer)(x4)

#     # Predicting CFFs
#     outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer, name='cff_output_layer')(x5)

#     # Concatenate input and output for further layers
#     total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)
    
#     # Assuming "TotalFLayer" is a custom layer (not defined here)
#     TotalF = TotalFLayer(name='TotalFLayer')(total_FInputs)
    
#     # Constructing the model
#     tfModel = tf.keras.Model(inputs=inputs, outputs=TotalF, name="Global_Fit_Model")
    
#     tfModel.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Define the learning rate
#         loss=tf.keras.losses.MeanSquaredError()
#     )
    
#     return tfModel

class PlotCustomizer:
    def __init__(
            self,
            axes: Axes,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "",
            zlabel: str = "",
            xlim = None,
            ylim = None,
            zlim = None,
            grid: bool = False):
        
        self._custom_rc_params = {
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
            'font.size': 14.,
            'pgf.rcfonts': False,  # Ensure it respects font settings
            'text.latex.preamble': r'\usepackage{lmodern}',
            # 'mathtext.fontset': 'dejavusans', # https://matplotlib.org/stable/gallery/text_labels_and_annotations/mathtext_fontfamily_example.html
            'xtick.direction': 'in',
            'xtick.major.size': 5,
            'xtick.major.width': 0.5,
            'xtick.minor.size': 2.5,
            'xtick.minor.width': 0.5,
            'xtick.minor.visible': True,
            'xtick.top': True,
            'ytick.direction': 'in',
            'ytick.major.size': 5,
            'ytick.major.width': 0.5,
            'ytick.minor.size': 2.5,
            'ytick.minor.width': 0.5,
            'ytick.minor.visible': True,
            'ytick.right': True,
        }

        self.axes_object = axes
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.grid = grid

        self._apply_customizations()

    def _apply_customizations(self):

        with rc_context(rc = self._custom_rc_params):

            # (1): Set the Title -- if it's not there, will set empty string:
            self.axes_object.set_title(self.title)

            # (2): Set the X-Label -- if it's not there, will set empty string:
            self.axes_object.set_xlabel(self.xlabel)

            # (3): Set the Y-Label -- if it's not there, will set empty string:
            self.axes_object.set_ylabel(self.ylabel)

            # (4): Set the X-Limit, if it's provided:
            if self.xlim:
                self.axes_object.set_xlim(self.xlim)

            # (5): Set the Y-Limit, if it's provided:
            if self.ylim:
                self.axes_object.set_ylim(self.ylim)

            # (6): Check if the Axes object is a 3D Plot that has 'set_zlabel' method:
            if hasattr(self.axes_object, 'set_zlabel'):

                # (6.1): If so, set the Z-Label -- if it's not there, will set empty string:
                self.axes_object.set_zlabel(self.zlabel)

            # (7): Check if the Axes object is 3D again and has a 'set_zlim' method:
            if self.zlim and hasattr(self.axes_object, 'set_zlim'):

                # (7.1): If so, set the Z-Limit, if it's provided:
                self.axes_object.set_zlim(self.zlim)

            # (8): Apply a grid on the plot according to a boolean flag:
            self.axes_object.grid(self.grid)

    def add_line_plot(self, x_data, y_data, label: str = "", color = None, linestyle = '-'):
        """
        Add a line plot to the Axes object:
        connects element-wise points of the two provided arrays.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        label: str

        color: str

        linestyle: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Just add the line plot:
            self.axes_object.plot(x_data, y_data, label = label, color = color, linestyle = linestyle)

            if label:
                self.axes_object.legend()

    def add_scatter_plot(self, x_data, y_data, radial_size: float = 1., label: str = "", color = None, marker = 'o'):
        """
        Add a scatter plot to the Axes object.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        radial_size: float
        
        label: str

        color: str 

        marker: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Add the scatter plot:
            self.axes_object.scatter(
                x_data,
                y_data,
                s = radial_size,
                label = label,
                color = color,
                marker = marker)

            if label:
                self.axes_object.legend()

SETTING_VERBOSE = True
SETTING_DEBUG = False
Learning_Rate = 0.001
EPOCHS = 2000
BATCH_SIZE = 20
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9
SETTING_DNN_TRAINING_VERBOSE = 2

callback_modify_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=modify_LR_factor, patience=modify_LR_patience, mode='auto')
callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience=EarlyStop_patience)

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


def generate_replica_data(df):
    """Generates a replica dataset by sampling F within sigmaF."""
    pseudodata_df = df.copy()

    # Ensure error values are positive
    pseudodata_df['sigmaF'] = np.abs(df['sigmaF'])

    # Generate normally distributed F values
    ReplicaF = np.random.normal(loc=df['F'], scale=pseudodata_df['sigmaF'])

    # Prevent negative values (ensuring no infinite loops)
    pseudodata_df['F'] = np.maximum(ReplicaF, 0)

    # Store original F values
    pseudodata_df['True_F'] = df['F']

    return pseudodata_df


def run_replica_method(number_of_replicas):

    if SETTING_VERBOSE:
        print(f"> Beginning Replica Method with {number_of_replicas} total Replicas...")

    for replica_index in range(number_of_replicas):

        initializer = tf.keras.initializers.RandomUniform(
            minval = -10.0,
            maxval = 10.0,
            seed = None)

        cross_section_inputs = Input(shape = (5,), name = 'input_layer')

        # q_squared, x_value, t_value, phi, k_value = tf.split(cross_section_inputs, num_or_size_splits = 5, axis = 1)

        q_squared = cross_section_inputs[:, 0:1]  # Extracting QQ from input
        x_value = cross_section_inputs[:, 1:2]  # Extracting x_b from input
        t_value = cross_section_inputs[:, 2:3]   # Extracting t from input
        phi = cross_section_inputs[:, 3:4]  # Extracting phi from input
        k = cross_section_inputs[:, 4:5]    # Extracting k from input

        kinematics = tf.keras.layers.concatenate([q_squared, x_value, t_value])
        
        # (3): Define the Model Architecture:
        x1 = tf.keras.layers.Dense(480, activation="relu6", kernel_initializer=initializer)(kinematics)
        x2 = tf.keras.layers.Dense(320, activation="relu6", kernel_initializer=initializer)(x1)
        x3 = tf.keras.layers.Dense(240, activation="relu6", kernel_initializer=initializer)(x2)
        x4 = tf.keras.layers.Dense(120, activation="relu6", kernel_initializer=initializer)(x3)
        x5 = tf.keras.layers.Dense(32, activation="relu6", kernel_initializer=initializer)(x4)
        
        output_y_value = Dense(4, activation = "linear", kernel_initializer = initializer, name = 'output_y_value')(x5)

        total_cross_section_inputs = tf.keras.layers.concatenate([cross_section_inputs, output_y_value], axis = 1)


        # (4): Define the model as as Keras Model:
        tensorflow_network = Model(inputs = cross_section_inputs, outputs = output_y_value, name = "basic_function_predictor")
        
        tensorflow_network.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [
                tf.keras.metrics.MeanSquaredError()
            ])
        
        tensorflow_network.summary()

        this_replica_data_set = data_file[data_file['set'] == kinematic_set_number].reset_index(drop = True)

        plt.figure(figsize=(10, 6))
        plt.errorbar(this_replica_data_set['phi_x'], this_replica_data_set['F'], this_replica_data_set['sigmaF'], fmt='o', label="True_F", color='red', markersize=5)
        plt.plot(this_replica_data_set['phi_x'], this_replica_data_set['F'], marker='o', linestyle='', label="Replica_F", color='blue')
        plt.title(f'F vs Phi for Kinematic Set {kinematic_set_number}')
        plt.xlabel(r'$\phi_x$')
        plt.ylabel('F')
        plt.legend(loc='best', fontsize='small')
        plt.savefig(f'F_vs_Phi_Kinematic_Set_{kinematic_set_number}_replica_{replica_index}.png')
        plt.close()

        pseudodata_dataframe = generate_replica_data(this_replica_data_set)

        training_x_data, testing_x_data, training_y_data, testing_y_data, training_y_error, testing_y_error = split_data(
            x_data = pseudodata_dataframe[['QQ', 'x_b', 't', 'phi_x', 'k']],
            y_data = pseudodata_dataframe['F'],
            y_error_data = pseudodata_dataframe['sigmaF'],
            split_percentage = 0.1)

        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        print(f"> Replica #{replica_index + 1} now running...")

        history_of_training_7 = tensorflow_network.fit(
            training_x_data,
            training_y_data,
            validation_data = (testing_x_data, testing_y_data),
            epochs = EPOCHS,
            callbacks = [
                callback_modify_learning_rate,
                callback_early_stop
            ],
            batch_size = BATCH_SIZE,
            verbose = SETTING_DNN_TRAINING_VERBOSE)
        
        training_loss_data_7 = history_of_training_7.history['loss']
        model_predictions_7 = tensorflow_network.predict(training_x_data)

        try:
            tensorflow_network.save(f"replica_number_{replica_index + 1}_v1.keras")
            print(f"> Saved replica!" )
        except Exception as ERROR:
             print(f"> Unable to save Replica model replica_number_{replica_index + 1}_v1.keras:\n> {ERROR}")
    
        end_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        print(f"> Replica job #{replica_index + 1} finished in {end_time_in_milliseconds - start_time_in_milliseconds}ms.")
        
        # (1): Set up the Figure instance
        figure_instance_nn_loss = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance_nn_loss = figure_instance_nn_loss.add_subplot(1, 1, 1)
        
        plot_customization_nn_loss = PlotCustomizer(
            axis_instance_nn_loss,
            title = r"Neural Network Loss",
            xlabel = r"Epoch",
            ylabel = r"Loss")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = training_loss_data_7,
            label = r'Training Loss',
            color = "black")
        
        # (1): Set up the Figure instance
        figure_instance_fitting = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance_fitting = figure_instance_fitting.add_subplot(1, 1, 1)
        
        plot_customization_data_comparison = PlotCustomizer(
            axis_instance_fitting,
            title = r"Fitting Procedure",
            xlabel = r"x",
            ylabel = r"f(x)")

        plot_customization_data_comparison.add_scatter_plot(
            x_data = training_x_data,
            y_data = training_y_data,
            label = r'Experimental Data',
            color = "red")
        
        plot_customization_data_comparison.add_scatter_plot(
            x_data = training_x_data,
            y_data = model_predictions_7,
            label = r'Model Predictions',
            color = "orange")
        
        figure_instance_nn_loss.savefig(f"loss_v{replica_index+1}_v1.png")
        figure_instance_fitting.savefig(f"fitting{replica_index+1}_v1.png")


DATA_FILE_NAME = "data.csv"
data_file = pd.read_csv(DATA_FILE_NAME)
data_file = data_file[data_file["F"] != 0]
kinematic_sets = sorted(data_file["set"].unique())

# Compute -t column
data_file['-t'] = -data_file['t']


# Function to compute density and scatter points
def density_scatter(x, y, ax, bins = 50, cmap='viridis'):
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xidx = np.digitize(x, xedges[:-1]) - 1
    yidx = np.digitize(y, yedges[:-1]) - 1
    density = counts[xidx, yidx]
    scatter = ax.scatter(x, y, c=density, cmap=cmap, s=10, alpha = 1.0, edgecolor = 'none')
    return scatter


# First plot: QQ vs x_b
fig, ax1 = plt.subplots(figsize=(8, 6))
density_scatter(data_file['x_b'], data_file['QQ'], ax1)
ax1.set_xlabel(r'$x_b$', fontsize=14)
ax1.set_ylabel(r'$Q^2 ext{ (GeV}^2ext{)}$', fontsize=14)
ax1.set_title(r'Density Plot of $Q^2$ vs $x_b$', fontsize=16)
ax1.set_xlim(0.0, 0.6)
ax1.set_ylim(0.5, 5.0)
ax1.grid(True, linestyle='--', linewidth=0.7)
plt.savefig('test_q2_vs_xb_v4.png')
plt.close()

# Second plot: -t vs x_b
fig, ax2 = plt.subplots(figsize=(8, 6))
density_scatter(data_file['x_b'], data_file['-t'], ax2)
ax2.set_xlabel(r'$x_b$', fontsize=14)
ax2.set_ylabel(r'$-t ext{ (GeV}^2ext{)}$', fontsize=14)
ax2.set_title(r'Density Plot of $-t$ vs $x_b$', fontsize=16)
ax2.set_xlim(0.0, 0.6)
ax2.grid(True, linestyle='--', linewidth=0.7)
plt.savefig('test_t_vs_q2_v4.png')
plt.close()

for kinematic_set_number in kinematic_sets:

    if SETTING_VERBOSE:
        print(f"> Now running kinematic set number {kinematic_set_number}...")

    run_replica_method(2)

    if SETTING_VERBOSE:
        print(f"> Finished running Replica Method on kinematic set number {kinematic_set_number}!")