from matplotlib.pyplot import Axes, rc_context

class PlotCustomizer:
    """Later description"""
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

    def add_line_plot(self, x_data, y_data, z_data=None, label: str = "", color=None, linestyle='-'):
        """
        Add a line plot to the Axes object:
        - In 2D: connects element-wise points of (x, y)
        - In 3D: connects element-wise points of (x, y, z)

        Parameters
        ----------
        x_data: array_like
            X-axis data points.

        y_data: array_like
            Y-axis data points.

        z_data: array_like, optional
            Z-axis data points (for 3D plotting).

        label: str
            Label for the line.

        color: str
            Color of the line.

        linestyle: str
            Line style (default is solid).
        """

        with rc_context(rc=self._custom_rc_params):
            if self.axes_object.name == '3d':  # Check if it's a 3D plot
                if z_data is None:
                    raise ValueError("For 3D plots, 'z_data' must be provided.")

                self.axes_object.plot(x_data, y_data, z_data, label=label, color=color, linestyle=linestyle)
            
            else:  # 2D case
                self.axes_object.plot(x_data, y_data, label=label, color=color, linestyle=linestyle)

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