import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, rc_context
plt.rcParams.update(plt.rcParamsDefault)
import math

_version_number = 1

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
    """FUCK IT"""
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
            'font.family': 'serif',
            'mathtext.fontset': 'cm', # https://matplotlib.org/stable/gallery/text_labels_and_annotations/mathtext_fontfamily_example.html
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

    def add_fill_between_plot(self, x_data, lower_y_data, upper_y_data, label: str = "", color = None, linestyle = '-', alpha = 1.0):
        """
        Add a line plot to the Axes object:
        connects element-wise points of the two provided arrays.

        Parameters
        ----------
        x_data: array_like
            
        lower_y_data: array_like

        upper_y_data: array_like

        label: str

        color: str

        linestyle: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Just add the line plot:
            self.axes_object.fill_between(
                x_data, 
                lower_y_data,
                upper_y_data, 
                label = label, 
                color = color, 
                linestyle = linestyle,
                alpha = alpha)

            if label:
                self.axes_object.legend()

    def add_scatter_plot(self, x_data, y_data, label: str = "", color = None, marker = 'o', markersize = None):
        """
        Add a scatter plot to the Axes object.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        label: str

        color: str |

        marker: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Add the scatter plot:
            self.axes_object.scatter(
                x = x_data,
                y = y_data,
                s = markersize,
                label = label,
                color = color,
                marker = marker)

            if label:
                self.axes_object.legend()

    def add_errorbar_plot(
            self,
            x_data,
            y_data,
            x_errorbars,
            y_errorbars,
            label: str = "",
            color = 'black',
            marker = 'o'):
        """
        Add a scatter plot with errorbars to the Axes object.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        x_errorbars: array_like
            
        y_errorbars: array_like

        label: str

        color: str |

        marker: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Add the errorbar plot:
            self.axes_object.errorbar(
                x = x_data,
                y = y_data, 
                yerr = y_errorbars,
                xerr = x_errorbars,
                label = label,
                color = color,
                marker = marker,
                linestyle = '', 
                markersize = 3.0,
                ecolor = 'black',
                elinewidth = 0.5,
                capsize = 1)

            if label:
                self.axes_object.legend()

    def add_bar_plot(self, x_data, y_data_heights=None, bins=None, label="", color=None, use_histogram=False):
        """
        Adds a bar plot to the existing axes.

        If `use_histogram=True`, `x_data` is treated as raw data, and histogram binning is applied.

        Parameters:
            x_data: If `use_histogram=False`, this is the x-coordinates for bars.
                    If `use_histogram=True`, this is the raw data to be binned.
            y_data_heights: Heights of bars (only used if `use_histogram=False`).
            bins: Number of bins (only used if `use_histogram=True`).
            label: Label for the legend.
            color: Color of the bars.
            use_histogram: If True, treat `x_data` as raw data and apply histogram binning.
        """

        with rc_context(rc=self._custom_rc_params):

            if use_histogram:
                # Compute histogram bin counts and bin edges
                y_data_heights, bin_edges = np.histogram(x_data, bins=bins)

                # Convert bin edges to bin centers for plotting
                x_data = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

            # (1): Add the bar plot:
            self.axes_object.bar(x_data, y_data_heights, label=label, color=color, edgecolor="black", alpha=0.7)

            if label:
                self.axes_object.legend()

    def add_3d_scatter_plot(self, x_data, y_data, z_data, color = None, marker = 'o'):

        with rc_context(rc = self._custom_rc_params):

            # (1): Plot points in R3:
            self.axes_object.scatter(x_data, y_data, z_data, color = color, marker = marker)

    def add_surface_plot(self, X, Y, Z, colormap: str ='viridis'):

        with rc_context(rc = self._custom_rc_params):

            # (1): Plot as surface in R3:
            self.axes_object.plot_surface(X, Y, Z, cmap = colormap, antialiased=False)

class F_calc:
    def __init__(self):
        self.module = BHDVCStf()

    def fn_1(self, kins, cffs):
        phi, QQ, x, t, k, F1, F2 = kins
        ReH, ReE, ReHtilde, c0fit = cffs
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.module.SetKinematics(QQ, x, t, k)
        P1, P2 = self.module.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.module.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.module.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return tf.get_static_value(f_pred)
    
class F1F2:
    def __init__(self):
        self.GM0 = 2.792847337

    def ffGE(self, t):
        GE = 1.0 / (1.0 + (-t / 0.710649)) / (1.0 + (-t / 0.710649))
        return GE

    def ffGM(self, t):
        shape = self.ffGE(t)
        return self.GM0 * shape

    def ffF2(self, t):
        f2 = (self.ffGM(t) - self.ffGE(t)) / (1.0 - t / (4.0 * 0.938272 * 0.938272))
        return f2

    def ffF1(self, t):
        f1 = self.ffGM(t) - self.ffF2(t)
        return f1

    def f1_f21(self, t):
        return self.ffF1(t), self.ffF2(t)
    
class BHDVCStf(object):

    def __init__(self):
        self.ALP_INV = tf.constant(137.0359998)  # 1 / Electromagnetic Fine Structure Constant
        self.PI = tf.constant(3.1415926535)
        self.RAD = tf.constant(self.PI / 180.)
        self.M = tf.constant(0.938272)  # Mass of the proton in GeV
        self.GeV2nb = tf.constant(.389379 * 1000000)  # Conversion from GeV to NanoBar
        self.M2 = tf.constant(0.938272 * 0.938272)  # Mass of the proton  squared in GeV

    @tf.function
    def SetKinematics(self, QQ, x, t, k):
        ee = 4. * self.M2 * x * x / QQ  # epsilon squared
        y = tf.sqrt(QQ) / (tf.sqrt(ee) * k)  # lepton energy fraction
        xi = x * (1. + t / 2. / QQ) / (2. - x + x * t / QQ);  # Generalized Bjorken variable
        Gamma = x * y * y / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / 8. / QQ / QQ / tf.sqrt(1. + ee)  # factor in front of the cross section, eq. (22)
        tmin = - QQ * (2. * (1. - x) * (1. - tf.sqrt(1. + ee)) + ee) / (4. * x * (1. - x) + ee)  # eq. (31)
        Ktilde_10 = tf.sqrt(tmin - t) * tf.sqrt((1. - x) * tf.sqrt(1. + ee) + ((t - tmin) * (ee + 4. * x * (1. - x)) / 4. / QQ)) * tf.sqrt(1. - y - y * y * ee / 4.) / tf.sqrt(1. - y + y * y * ee / 4.)  # K tilde from 2010 paper
        K = tf.sqrt(1. - y + y * y * ee / 4.) * Ktilde_10 / tf.sqrt(QQ)
        return ee, y, xi, Gamma, tmin, Ktilde_10, K

    @tf.function
    def BHLeptonPropagators(self, phi, QQ, x, t, ee, y, K):
        # KD 4-vector product (phi-dependent)
        KD = - QQ / (2. * y * (1. + ee)) * (1. + 2. * K * tf.cos(self.PI - (phi * self.RAD)) - t / QQ * (1. - x * (2. - y) + y * ee / 2.) + y * ee / 2.)  # eq. (29)

        # lepton BH propagators P1 and P2 (contaminating phi-dependence)
        P1 = 1. + 2. * KD / QQ
        P2 = t / QQ - 2. * KD / QQ
        return P1, P2

    @tf.function
    def BHUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K):

        # BH utorch.larized Fourier harmonics eqs. (35 - 37)
        phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K = [tf.cast(i, np.float32) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K]]
        c0_BH = 8. * K * K * ((2. + 3. * ee) * (QQ / t) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * x * x * (F1 + F2) * (F1 + F2)) + (2. - y) * (2. - y) * ((2. + ee) * (
                    (4. * x * x * self.M2 / t) * (1. + t / QQ) * (
                        1. + t / QQ)
                        + 4. * (1. - x) * (1. + x * t / QQ)) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 4. * x * x * (x + (1. - x + ee / 2.) * (1. - t / QQ) * (1. - t / QQ) - x * (1. - 2. * x) * t * t / (QQ * QQ)) * (F1 + F2) * (F1 + F2)) + 8. * (
                                 1. + ee) * (1. - y - ee * y * y / 4.) * (
                                 2. * ee * (1. - t / (4. * self.M2)) * (
                                     F1 * F1 - F2 * F2 * t / (4. * self.M2)) - x * x * (
                                             1. - t / QQ) * (1. - t / QQ) * (F1 + F2) * (F1 + F2))

        c1_BH = 8. * K * (2. - y) * (
                    (4. * x * x * self.M2 / t - 2. * x - ee) * (
                        F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * x * x * (
                                1. - (1. - 2. * x) * t / QQ) * (F1 + F2) * (F1 + F2))

        c2_BH = 8. * x * x * K * K * (
                    (4. * self.M2 / t) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * (F1 + F2) * (
                        F1 + F2))

        # BH squared amplitude eq (25) divided by e^6
        Amp2_BH = 1. / (x * x * y * y * (1. + ee) * (
                    1. + ee) * t * P1 * P2) * (c0_BH + c1_BH * tf.cos(
            self.PI - (phi * self.RAD)) + c2_BH * tf.cos(2. * (self.PI - (phi * self.RAD))))

        Amp2_BH = self.GeV2nb * Amp2_BH  # convertion to nb

        return Gamma * Amp2_BH

    @tf.function
    def IUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, twist, tmin, xi, Ktilde_10):
        phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10 = [tf.cast(i, np.float32) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10]]
        # Get BH propagators and set the kinematics
        self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)

        # Get A_UU_I, B_UU_I and C_UU_I interference coefficients
        A_U_I, B_U_I, C_U_I = self.ABC_UU_I_10(phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10)

        # BH-DVCS interference squared amplitude
        I_10 = 1. / (x * y * y * y * t * P1 * P2) * (
                    A_U_I * (F1 * ReH - t / 4. / self.M2 * F2 * ReE) + B_U_I * (F1 + F2) * (
                        ReH + ReE) + C_U_I * (F1 + F2) * ReHtilde)

        I_10 = self.GeV2nb * I_10  # convertion to nb

        return Gamma * I_10

    @tf.function
    def ABC_UU_I_10(self, phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10):  # Get A_UU_I, B_UU_I and C_UU_I interference coefficients BKM10

        if twist == "t2":
            f = 0  # F_eff = 0 ( pure twist 2)
        if twist == "t3":
            f = - 2. * xi / (1. + xi)
        if twist == "t3ww":
            f = 2. / (1. + xi)

        # Interference coefficients  (BKM10 Appendix A.1)
        # n = 0 -----------------------------------------
        # helicity - conserving (F)
        C_110 = - 4. * (2. - y) * (1. + tf.sqrt(1 + ee)) / tf.pow((1. + ee), 2) * (
                    Ktilde_10 * Ktilde_10 * (2. - y) * (2. - y) / QQ / tf.sqrt(1 + ee)
                    + t / QQ * (1. - y - ee / 4. * y * y) * (2. - x) * (1. + (
                        2. * x * (2. - x + (tf.sqrt(
                    1. + ee) - 1.) / 2. + ee / 2. / x) * t / QQ + ee) / (2. - x) / (
                                                                                                                       1. + tf.sqrt(
                                                                                                                   1. + ee))))
        C_110_V = 8. * (2. - y) / tf.pow((1. + ee), 2) * x * t / QQ * (
                    (2. - y) * (2. - y) / tf.sqrt(1. + ee) * Ktilde_10 * Ktilde_10 / QQ
                    + (1. - y - ee / 4. * y * y) * (1. + tf.sqrt(1. + ee)) / 2. * (
                                1. + t / QQ) * (1. + (tf.sqrt(1. + ee) - 1. + 2. * x) / (
                        1. + tf.sqrt(1. + ee)) * t / QQ))
        C_110_A = 8. * (2. - y) / tf.pow((1. + ee), 2) * t / QQ * (
                    (2. - y) * (2. - y) / tf.sqrt(
                1. + ee) * Ktilde_10 * Ktilde_10 / QQ * (
                                1. + tf.sqrt(1. + ee) - 2. * x) / 2.
                    + (1. - y - ee / 4. * y * y) * ((1. + tf.sqrt(1. + ee)) / 2. * (
                        1. + tf.sqrt(1. + ee) - x + (
                            tf.sqrt(1. + ee) - 1. + x * (3. + tf.sqrt(1. + ee) - 2. * x) / (
                                1. + tf.sqrt(1. + ee)))
                        * t / QQ) - 2. * Ktilde_10 * Ktilde_10 / QQ))
        # helicity - changing (F_eff)
        C_010 = 12. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 ee + (2. - 6. * x - ee) / 3. * t / QQ)
        C_010_V = 24. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * x * t / QQ * (
                                   1. - (1. - 2. * x) * t / QQ)
        C_010_A = 4. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * t / QQ * (
                                   8. - 6. * x + 5. * ee) * (
                                   1. - t / QQ * ((2. - 12 * x * (1. - x) - ee)
                                                            / (8. - 6. * x + 5. * ee)))
        # n = 1 -----------------------------------------
        # helicity - conserving (F)
        C_111 = -16. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * ((1. + (1. - x) * (tf.sqrt(
            1 + ee) - 1.) / 2. / x + ee / 4. / x) * x * t / QQ - 3. * ee / 4.) - 4. * K * (
                                 2. - 2. * y + y * y + ee / 2. * y * y) * (
                                 1. + tf.sqrt(1 + ee) - ee) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 1. - (1. - 3. * x) * t / QQ + (
                                     1. - tf.sqrt(1 + ee) + 3. * ee) / (
                                             1. + tf.sqrt(1 + ee) - ee) * x * t / QQ)
        C_111_V = 16. * K / tf.pow(tf.sqrt(1. + ee), 5) * x * t / QQ * (
                    (2. - y) * (2. - y) * (1. - (1. - 2. * x) * t / QQ) + (
                        1. - y - ee / 4. * y * y)
                    * (1. + tf.sqrt(1. + ee) - 2. * x) / 2. * (t - tmin) / QQ)
        C_111_A = -16. * K / tf.pow((1. + ee), 2) * t / QQ * (
                    (1. - y - ee / 4. * y * y) * (1. - (1. - 2. * x) * t / QQ + (
                        4. * x * (1. - x) + ee) / 4. / tf.sqrt(1. + ee) * (
                                                                                  t - tmin) / QQ)
                    - tf.pow((2. - y), 2) * (
                                1. - x / 2. + (1. + tf.sqrt(1. + ee) - 2. * x) / 4. * (
                                    1. - t / QQ) + (4. * x * (1. - x) + ee) / 2. / tf.sqrt(
                            1. + ee) * (t - tmin) / QQ))
        # helicity - changing (F_eff)
        C_011 = 8. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            (1. + ee), 2) * (tf.pow((2. - y), 2) * (t - tmin) / QQ * (
                    1. - x + ((1. - x) * x + ee / 4.) / tf.sqrt(1. + ee) * (
                        t - tmin) / QQ)
                                  + (1. - y - ee / 4. * y * y) / tf.sqrt(1 + ee) * (
                                              1. - (1. - 2. * x) * t / QQ) * (
                                              ee - 2. * (1. + ee / 2. / x) * x * t / QQ))
        C_011_V = 16. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * x * t / QQ * (
                                   tf.pow(Ktilde_10 * (2. - y), 2) / QQ + tf.pow(
                               (1. - (1. - 2. * x) * t / QQ), 2) * (
                                               1. - y - ee / 4. * y * y))
        C_011_A = 8. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * t / QQ * (
                                   tf.pow(Ktilde_10 * (2. - y), 2) * (1. - 2. * x) / QQ + (
                                       1. - (1. - 2. * x) * t / QQ)
                                   * (1. - y - ee / 4. * y * y) * (
                                               4. - 2. * x + 3. * ee + t / QQ * (
                                                   4. * x * (1. - x) + ee)))
        # n = 2 -----------------------------------------
        # helicity - conserving (F)
        C_112 = 8. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                     2) * (
                                 2. * ee / tf.sqrt(1. + ee) / (1. + tf.sqrt(1. + ee)) * tf.pow(
                             Ktilde_10, 2) / QQ + x * t * (
                                             t - tmin) / QQ / QQ * (1. - x - (
                                     tf.sqrt(1. + ee) - 1.) / 2. + ee / 2. / x))
        C_112_V = 8. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                       2) * x * t / QQ * (
                                   4. * tf.pow(Ktilde_10, 2) / tf.sqrt(1. + ee) / QQ + (
                                       1. + tf.sqrt(1. + ee) - 2. * x) / 2. * (1. + t / QQ) * (
                                               t - tmin) / QQ)
        C_112_A = 4. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                       2) * t / QQ * (
                                   4. * (1. - 2. * x) * tf.pow(Ktilde_10, 2) / tf.sqrt(
                               1. + ee) / QQ - (3. - tf.sqrt(
                               1. + ee) - 2. * x + ee / x) * x * (
                                               t - tmin) / QQ)
        # helicity - changing (F_eff)
        C_012 = -8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 1. + ee / 2.) * (
                                 1. + (1. + ee / 2. / x) / (1. + ee / 2.) * x * t / QQ)
        C_012_V = 8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * x * t / QQ * (
                                   1. - (1. - 2. * x) * t / QQ)
        C_012_A = 8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow((1. + ee), 2) * t / QQ * (
                                   1. - x + (t - tmin) / 2. / QQ * (
                                       4. * x * (1. - x) + ee) / tf.sqrt(1. + ee))
        # n = 3 -----------------------------------------
        # helicity - conserving (F)
        C_113 = -8. * K * (1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                                               5) * (
                                 tf.sqrt(1. + ee) - 1.) * (
                                 (1. - x) * t / QQ + (tf.sqrt(1. + ee) - 1.) / 2. * (
                                     1. + t / QQ))
        C_113_V = -8. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * x * t / QQ * (tf.sqrt(1. + ee) - 1. + (
                    1. + tf.sqrt(1. + ee) - 2. * x) * t / QQ)
        C_113_A = 16. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * t * (t - tmin) / QQ / QQ * (
                                   x * (1. - x) + ee / 4.)

        # A_U_I, B_U_I and C_U_I
        A_U_I = C_110 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
            QQ) * f * C_010 + (C_111 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
            QQ) * f * C_011) * tf.cos(self.PI - (phi * self.RAD)) + (
                                 C_112 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                             QQ) * f * C_012) * tf.cos(
            2. * (self.PI - (phi * self.RAD))) + C_113 * tf.cos(3. * (self.PI - (phi * self.RAD)))
        B_U_I = xi / (1. + t / 2. / QQ) * (
                    C_110_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * C_010_V + (
                                C_111_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * C_011_V) * tf.cos(self.PI - (phi * self.RAD)) + (
                                C_112_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * C_012_V) * tf.cos(
                2. * (self.PI - (phi * self.RAD))) + C_113_V * tf.cos(3. * (self.PI - (phi * self.RAD))))
        C_U_I = xi / (1. + t / 2. / QQ) * (
                    C_110 + C_110_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * (C_010 + C_010_A) + (
                                C_111 + C_111_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * (C_011 + C_011_A)) * tf.cos(self.PI - (phi * self.RAD)) + (
                                C_112 + C_112_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * (C_012 + C_012_A)) * tf.cos(
                2. * (self.PI - (phi * self.RAD))) + (C_113 + C_113_A) * tf.cos(
                3. * (self.PI - (phi * self.RAD))))

        return A_U_I, B_U_I, C_U_I

    @tf.function
    def curve_fit(self, kins, cffs):
        calc = F1F2()
        QQ, x, t, phi, k = tf.split(kins, num_or_size_splits=5, axis=1)
        F1, F2 = calc.f1_f21(t) # calculating F1 and F2 using passed data as opposed to passing in F1 and F2
        ReH, ReE, ReHtilde, c0fit = tf.split(cffs, num_or_size_splits=4, axis=1)  # output of network
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.SetKinematics(QQ, x, t, k)
        P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return f_pred
    
class TotalFLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TotalFLayer, self).__init__(**kwargs)  # Pass kwargs to the superclass
        self.f = BHDVCStf()

    def call(self, inputs):
        return self.f.curve_fit(inputs[:, 0:5], inputs[:, 5:9]) # QQ, x, t, phi, k, cff1, cff2, cff3, cff4
    
SETTING_VERBOSE = True
SETTING_DEBUG = True
Learning_Rate = 0.001
EPOCHS = 300
BATCH_SIZE = 20
EarlyStop_patience = 1000
modify_LR_patience = 400
modify_LR_factor = 0.9
SETTING_DNN_TRAINING_VERBOSE = 1

callback_modify_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=modify_LR_factor, patience=modify_LR_patience, mode='auto')
callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience=EarlyStop_patience)

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

    # (4): Whatever the fuck this means:
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

def generate_replica_data(pandas_dataframe: pd.DataFrame, mean_value_column_name: str, stddev_column_name: str):
    """Generates a replica dataset by sampling F within sigmaF."""
    pseudodata_dataframe = pandas_dataframe.copy()

    # Ensure error values are positive
    pseudodata_dataframe[stddev_column_name] = np.abs(pandas_dataframe[stddev_column_name])

    # Generate normally distributed F values
    replica_cross_section_sample = np.random.normal(loc = pandas_dataframe[mean_value_column_name], scale = pseudodata_dataframe[stddev_column_name])

    # Prevent negative values (ensuring no infinite loops)
    pseudodata_dataframe[mean_value_column_name] = np.maximum(replica_cross_section_sample, 0)

    # Store original F values
    pseudodata_dataframe['True_F'] = pandas_dataframe[mean_value_column_name]

    return pseudodata_dataframe

def build_global_fitting_dnn():
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

def circle_dnn():
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

    output_y_value = Dense(1, activation = "linear", kernel_initializer = initializer, name = "cff_output_layer")(x5)

    tensorflow_network = Model(inputs = circle_function_input, outputs = output_y_value, name = "spherical_function_fitter")

    return tensorflow_network

def build_dnn():
    """Creates and returns a fresh instance of the neural network model."""
    initializer = tf.keras.initializers.RandomUniform(
        minval=-10.0,
        maxval=10.0,
        seed = None)

    cross_section_inputs = Input(shape=(5,), name='input_layer')

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

    output_y_value = Dense(4, activation="linear", kernel_initializer = initializer, name='cff_output_layer')(x5)

    total_cross_section_inputs = tf.keras.layers.concatenate([cross_section_inputs, output_y_value], axis = 1)

    tensorflow_network = Model(inputs = cross_section_inputs, outputs=output_y_value, name="tfmodel")

    return tensorflow_network

def run_replica_method(number_of_replicas, model_builder, data_file, kinematic_set_number):

    if SETTING_VERBOSE:
        print(f"> Beginning Replica Method with {number_of_replicas} total Replicas...")

    for replica_index in range(number_of_replicas):

        if SETTING_VERBOSE:
            print(f"> Now initializing replica #{replica_index + 1}...")

        tensorflow_network = model_builder()
        
        tensorflow_network.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [
                tf.keras.metrics.MeanSquaredError()
            ])
        
        # tensorflow_network.summary()

        this_replica_data_set = data_file[data_file['set'] == kinematic_set_number].reset_index(drop = True)

        pseudodata_dataframe = generate_replica_data(
            pandas_dataframe = this_replica_data_set,
            mean_value_column_name = 'F',
            stddev_column_name = 'sigmaF')

        training_x_data, testing_x_data, training_y_data, testing_y_data, training_y_error, testing_y_error = split_data(
            x_data = pseudodata_dataframe[['QQ', 'x_b', 't', 'phi_x', 'k']],
            y_data = pseudodata_dataframe['F'],
            y_error_data = pseudodata_dataframe['sigmaF'],
            split_percentage = 0.1)
        
        print(training_x_data)
        
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
        
        training_loss_data = history_of_training_7.history['loss']
        validation_loss_data = history_of_training_7.history['val_loss']

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

# Function to compute density and scatter points
def density_scatter(x, y, ax, bins = 50, cmap='viridis'):
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xidx = np.digitize(x, xedges[:-1]) - 1
    yidx = np.digitize(y, yedges[:-1]) - 1
    density = counts[xidx, yidx]
    scatter = ax.scatter(x, y, c = density, cmap = cmap, s = 10, alpha = 1.0, edgecolor = 'none')
    return scatter

def run():

    tf.config.set_visible_devices([],'GPU')

    try:
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
        title = r"[Experimental] Kinematic Phase Spac",
        xlabel = r"$x_{{B}}$",
        ylabel = r"$Q^{{2}}$",
        zlabel = r"$-t$",
        grid = True)
    plot_customization_data_comparison.add_3d_scatter_plot(
        x_data = data_file['x_b'],
        y_data = data_file['QQ'],
        z_data = -data_file['t'],
        color = 'red',
        marker = '.')
    figure_instance_kinematic_phase_space.savefig(f"phase_space_v{_version_number}.png")
    plt.close()
    
    # for kinematic_set_number in kinematic_sets:
    for kinematic_set_number in [1.0]:

        if SETTING_VERBOSE:
            print(f"> Now running kinematic set number {kinematic_set_number}...")

        run_replica_method(
            number_of_replicas = 5,
            model_builder = build_dnn,
            data_file = data_file,
            kinematic_set_number = kinematic_set_number)

        if SETTING_VERBOSE:
            print(f"> Finished running Replica Method on kinematic set number {kinematic_set_number}!")

    available_kinematic_sets = [1, 2, 3, 4]
    for available_kinematic_set in available_kinematic_sets:

        import os

        if SETTING_VERBOSE:
            print(f"> Now generating .csv files for kinematic set #{available_kinematic_set}...")

        # model_paths = [os.path.join(os.getcwd(), file) for file in os.listdir(os.getcwd()) if file.endswith(f"v{_version_number}.keras")]
        # model_paths = os.getcwd()

        try:
            model_paths = [os.path.join(os.getcwd(), file) for file in os.listdir(os.getcwd()) if file.endswith(f"v{_version_number}.keras")]
            if SETTING_DEBUG:
                print(f"> Successfully  captured {len(model_paths)} in list for iteration.")

        except Exception as error:
            print(f" Error in capturing replicas in list:\n> {error}")

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
        axis_instance_cff_h_histogram = figure_cff_real_h_histogram.add_subplot(1, 1, 1)
        plot_customization_cff_h_histogram = PlotCustomizer(
            axis_instance_cff_h_histogram,
            title = r"Predictions for $Re \left(H \right)$")
        plot_customization_cff_h_histogram.add_bar_plot(
            x_data = np.array(predictions_for_cffs)[:, :, 1].T.flatten(),
            bins = 20,
            label = "Histogram Bars",
            color = "lightblue",
            use_histogram = True)
        figure_cff_real_h_histogram.savefig(f"shit_v{_version_number}.png")
        plt.close()
        # cff_real_E_histogram = plt.figure(figsize = (18, 6))
        # cff_real_Ht_histogram = plt.figure(figsize = (18, 6))
        # cff_dvcs_histogram = plt.figure(figsize = (18, 6))
    
        

            # # (1): Set up the Figure instance
            # figure_instance_fitting = plt.figure(figsize = (18, 6))

            # # (2): Add an Axes Object:
            # axis_instance_fitting = figure_instance_fitting.add_subplot(1, 1, 1)
            
            # plot_customization_data_comparison = PlotCustomizer(
            #     axis_instance_fitting,
            #     title = r"Replica Fitting",
            #     xlabel = r"\phi",
            #     ylabel = r"\sigma")
            
            # plot_customization_predictions.add_errorbar_plot(
            #     x_data = this_replica_data_set['phi_x'],
            #     y_data = this_replica_data_set['F'],
            #     x_errorbars = np.zeros(this_replica_data_set['sigmaF'].shape),
            #     y_errorbars = this_replica_data_set['sigmaF'],
            #     label = r'Raw Data',
            #     color = "black")
            
            # plot_customization_data_comparison.add_scatter_plot(
            #     x_data = np.linspace(0, 361, 1),
            #     y_data = model_predictions,
            #     label = r'Model Predictions',
            #     color = "orange")
            
            # figure_instance_fitting.savefig(f"fitting{replica_index+1}_v{_version_number}.png")

    import pysr
    from pysr import PySRRegressor

    cross_section_model = tf.keras.models.load_model(f"replica_number_1_v{_version_number}.keras")
    cff_tf_model = tf.keras.Model(
                inputs = cross_section_model.input,
                outputs = cross_section_model.get_layer('cff_output_layer').output)
            
    if SETTING_DEBUG:
        print("> Successfully CFF submodel!")

    X_train = np.column_stack([
        data_file['k'],
        data_file['QQ'],
        data_file['x_b'],
        data_file['t'],
        data_file['phi_x']
        ])
    Y_train_cffs = cff_tf_model.predict(prediction_inputs)
    Y_train_cross_section = cross_section_model.predict(prediction_inputs)   

    cff_model = PySRRegressor(
    niterations=1000,  # Adjust based on complexity
    binary_operators=["+", "-", "*", "/"],  # Allowed operations
    unary_operators=["exp", "log", "sin", "cos"],  # Allowed functions
    extra_sympy_mappings={},  # Custom functions if needed
    model_selection="best",  # Choose the simplest best-performing model
    progress=True,)

    cff_model.fit(X_train, Y_train_cffs)  # Fit symbolic regression model

    X_train_extended = np.hstack([X_train, Y_train_cffs])  # Append CFFs as additional inputs

    cross_section_model = PySRRegressor(
        niterations=1000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sin", "cos"],
        model_selection="best",
        progress=True,
    )

    cross_section_model.fit(X_train_extended, Y_train_cross_section)

    print("Discovered formulas for CFFs:")
    print(cff_model)

    print("\nDiscovered formula for the cross section:")
    print(cross_section_model)
    
    # Predict CFFs
    Y_pred_cffs = cff_model.predict(X_train)

    # Predict cross section
    Y_pred_cross_section = cross_section_model.predict(X_train_extended)

    # Plot CFFs
    for i in range(4):
        plt.figure(figsize=(6,4))
        plt.scatter(Y_train_cffs[:, i], Y_pred_cffs[:, i], alpha=0.5)
        plt.xlabel("True CFF Value")
        plt.ylabel("Predicted CFF Value")
        plt.title(f"CFF {i+1}: True vs. Predicted")
        plt.plot([min(Y_train_cffs[:, i]), max(Y_train_cffs[:, i])], 
                [min(Y_train_cffs[:, i]), max(Y_train_cffs[:, i])], 'r--')
        plt.show()

    # Plot Cross Section
    plt.figure(figsize=(6,4))
    plt.scatter(Y_train_cross_section, Y_pred_cross_section, alpha=0.5)
    plt.xlabel("True Cross Section")
    plt.ylabel("Predicted Cross Section")
    plt.title("Cross Section: True vs. Predicted")
    plt.plot([min(Y_train_cross_section), max(Y_train_cross_section)], 
            [min(Y_train_cross_section), max(Y_train_cross_section)], 'r--')
    plt.show()
    

if __name__ == "__main__":

    run()