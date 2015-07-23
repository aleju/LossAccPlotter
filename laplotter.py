from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from six.moves import range


class LossAccPlot(object):
    """Class to plot training and validation loss and accuracy.
    """
    def __init__(self,
                 save_to_filepath=None,
                 show_regressions=True,
                 show_averages=True,
                 show_loss_plot=True,
                 show_acc_plot=True,
                 show_plot_window=True):
        """Constructs the plotter.
        Args:
            save_to_filepath: The filepath to a file at which the plot
                is ought to be saved, e.g. "/tmp/last_plot.png". Set this value
                to None if you don't want to save the plot.
            show_plot_window: Whether to show the plot in a window (True)
                or hide it (False). Hiding it makes only sense if you
                set save_to_filepath.
            linestyles: List of two string values containing the stylings
                of the chart lines. The first value is for the training
                line, the second for the validation line. Loss and accuracy
                charts will both use that styling.
            linestyles_first_epoch: Different stylings for the chart lines
                for the very first epoch (no two points yet to draw a line).
            show_regression: Whether or not to show a regression, indicating
                where each line might end up in the future.
            poly_forward_perc: Percentage value (e.g. 0.1 = 10%) indicating
                for how far in the future each regression line will be
                calculated. The percentage is relative to the current epoch.
                E.g. if epoch is 100 and this value is 0.2, then the regression
                will be calculated for 20 values in the future.
            poly_backward_perc: Similar to poly_forward_perc. Percentage of
                the data basis to use in order to calculate the regression.
                E.g. if epoch is 100 and this value is 0.2, then the last
                20 values will be used to predict the future values.
            poly_n_forward_min: Minimum value for how far in the future
                the regression values will be predicted for each line.
                E.g. 10 means that there will always be at least 10 predicted
                values, even for e.g. epoch 5.
            poly_n_backward_min: Similar to poly_n_forward_min. Minimum
                epochs to use backwards for predicting future values.
            poly_degree: Degree of the polynomial to use when predicting
                future values. Should usually be 1.
        """
        assert show_loss_plot or show_acc_plot
        assert save_to_filepath is not None or show_plot_window

        self.linestyles = linestyles
        self.linestyles_first_epoch = linestyles_first_epoch
        self.show_regressions = show_regressions
        self.show_averages = show_averages
        self.show_loss_plot = show_loss_plot
        self.show_acc_plot = show_acc_plot
        self.show_plot_window = show_plot_window
        self.save_to_filepath = save_to_filepath

        self.poly_forward_perc=0.1,
        self.poly_backward_perc=0.2,
        self.poly_n_forward_min=5,
        self.poly_n_backward_min=10,
        self.poly_degree=1

        self.linestyles = {
            "loss_train": "r-",
            "loss_train_regression": "r:",
            "loss_val": "b-",
            "loss_val_regression": "b:",
            "acc_train": "r-",
            "acc_train_regression": "r:",
            "acc_val": "b-",
            "acc_val_regression": "b:"
        }
        # different linestyles for the first epoch, because there will be only
        # one value available => no line can be drawn
        self.linestyles_one_value = {
            "loss_train": "rs-",
            "loss_train_regression": "r:",
            "loss_val": "b^-",
            "loss_val_regression": "b^:",
            "acc_train": "rs-",
            "acc_train_regression": "r:",
            "acc_val": "b^-",
            "acc_val_regression": "b^:"
        }

        # ----
        # Initialize plots
        # ----
        if show_loss_plot and show_acc_plot:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24, 8))
            self.fig = fig
            self.ax_loss = ax1
            self.ax_acc = ax2
        else:
            fig, (ax,) = plt.subplots(ncols=1, figsize=(12, 8))
            self.fig = fig
            self.ax_loss = ax if show_loss_plot else None
            self.ax_acc = ax if show_acc_plot else None

        # set_position is neccessary here in order to place the legend properly
        for ax in [self.ax_loss, self.ax_acc]:
            if ax is not None:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.9])

        self.values_loss_train_x = []
        self.values_loss_val_x = []
        self.values_acc_train_x = []
        self.values_acc_val_x = []
        self.values_loss_train_y = []
        self.values_loss_val_y = []
        self.values_acc_train_y = []
        self.values_acc_val_y = []

    def add_values(self, x_index, loss_train=None, loss_val=None, acc_train=None,
                   acc_val=None, redraw=True):
        if loss_train is not None:
            self.values_loss_train_x.append(x_index)
            self.values_loss_train_y.append(loss_train)
        if loss_val is not None:
            self.values_loss_val_x.append(x_index)
            self.values_loss_val_y.append(loss_val)
        if acc_train is not None:
            self.values_acc_train_x.append(x_index)
            self.values_acc_train_y.append(acc_train)
        if acc_val is not None:
            self.values_acc_val_x.append(x_index)
            self.values_acc_val_y.append(acc_val)

        if redraw:
            self._redraw()
            if self.save_to_filepath:
                self._save_plot(self.save_to_filepath)

    def _save_plot(self, filepath):
        """Saves the current plot to a file.
        Args:
            filepath: The path to the file, e.g. "/tmp/last_plot.png".
        """
        self.fig.savefig(filepath)

    def _redraw(self):
        """Redraws the plot with new values.
        Args:
            epoch: The index of the current epoch, starting at 0.
            train_loss: All of the training loss values of each
                epoch (list of floats).
            train_acc: All of the training accuracy values of each
                epoch (list of floats).
            val_loss: All of the validation loss values of each
                epoch (list of floats).
            val_acc: All of the validation accuracy values of each
                epoch (list of floats).
        """

        if self.first_redraw and self.show_plot_window:
            plt.figure(self.fig.number)
            plt.show(block=False)
            #plt.draw()

        ax1 = self.ax_loss
        ax2 = self.ax_acc

        # List of each epoch (x-axis)
        epochs = list(range(0, epoch+1))

        # Clear loss and accuracy charts
        if ax1:
            ax1.clear()
        if ax2:
            ax2.clear()

        # Set titles of charts (at the top)
        if ax1:
            ax1.set_title("loss")
        if ax2:
            ax2.set_title("accuracy")

        # Set the styles of the lines used in the charts
        # Different line style for epochs after the  first one, because
        # the very first epoch has only one data point and therefore no line
        # and would be invisible without the changed style.
        ls_loss_train = self.linestyles["loss_train"]
        ls_loss_val = self.linestyles["loss_val"]
        ls_acc_train = self.linestyles["acc_train"]
        ls_acc_val = self.linestyles["acc_val"]
        if len(self.values_loss_train_x) == 1:
            ls_loss_train = self.linestyles_one_value["loss_train"]
        if len(self.values_loss_val_x) == 1:
            ls_loss_val = self.linestyles_one_value["loss_val"]
        if len(self.values_acc_train_x) == 1:
            ls_acc_train = self.linestyles_one_value["acc_train"]
        if len(self.values_acc_val_x) == 1:
            ls_acc_val = self.linestyles_one_value["acc_val"]

        # Plot the lines
        if ax1:
            ax1.plot(self.values_loss_train_x, self.values_loss_train_y,
                     ls_loss_train, label="loss train")
            ax1.plot(self.values_loss_val_x, self.values_loss_val_y,
                     ls_loss_val, label="loss val.")
        if ax2:
            ax2.plot(self.values_acc_train_x, self.values_acc_train_y,
                     ls_acc_train, label="acc. train")
            ax2.plot(self.values_acc_val_x, self.values_acc_val_y,
                     ls_acc_val, label="acc. val.")

        if self.show_regressions:
            # Compute the regression lines for the n_forward future epochs.
            # n_forward is calculated relative to the current epoch
            # (e.g. at epoch 100 compute 10 next, at 200 the 20 next ones...).
            n_forward = int(max((epoch+1)*self.poly_forward_perc,
                                self.poly_forward_min))

            # Compute regression lines based on n_backwards epochs
            # in the past, e.g. based on the last 10 values.
            # n_backwards is calculated relative to the current epoch
            # (e.g. at epoch 100 compute based on the last 10 values,
            # at 200 based on the last 20 values...).
            n_backwards = int(max((epoch+1)*self.poly_backward_perc,
                                  self.poly_backward_min))

            # List of epochs for which to estimate/predict the likely value.
            # (epoch..epoch+n_forward instead of epoch+1..epoch+n_forward
            # so that the regression line is better connected to the line its
            # based on (no obvious gap).)
            future_epochs = [i for i in range(epoch, epoch + n_forward)]

            self.plot_regression_line(ax1, train_loss, epochs, future_epochs,
                                      n_backwards, linestyles[2],
                                      'train loss regression')
            self.plot_regression_line(ax1, val_loss, epochs, future_epochs,
                                      n_backwards, linestyles[3],
                                      'val loss regression')
            self.plot_regression_line(ax2, train_acc, epochs, future_epochs,
                                      n_backwards, linestyles[2],
                                      'train acc regression')
            self.plot_regression_line(ax2, val_acc, epochs, future_epochs,
                                      n_backwards, linestyles[3],
                                      'val acc regression')

        # Add legend (below chart)
        if ax1:
            ax1.legend(["loss train", "loss val.",
                        "loss train regression", "loss val. regression"],
                       bbox_to_anchor=(0.7, -0.08), ncol=2)
        if ax2:
            ax2.legend(["acc. train", "acc. val.",
                        "acc. train regression", "acc. val. regression"],
                       bbox_to_anchor=(0.7, -0.08), ncol=2)

        # Labels for x and y axis
        if ax1:
            ax1.set_ylabel("loss")
            ax1.set_xlabel("epoch")
        if ax2:
            ax2.set_ylabel("accuracy")
            ax2.set_xlabel("epoch")

        # Show a grid in both charts
        if ax1:
            ax1.grid(True)
        if ax2:
            ax2.grid(True)

    def plot_regression_line(self, plot_ax, data, epochs, future_epochs,
                             n_backwards, linestyle, label):
        """Calculates and plots a regression line.
        Args:
            plot_ax: The ax on which to plot the line.
            data: The data used to perform the regression
                (e.g. training loss values).
            epochs: List of all epochs (0, 1, 2, ...).
            future_epochs: List of the future epochs for which values are
                ought to be predicted.
            n_backwards: How far back to go in time (in epochs) in order
                to compute the regression. (E.g. 10 = calculate it on the
                last 10 values max.)
            linestyle: Linestyle of the regression line.
            label: Label of the regression line.
        """
        # dont try to draw anything if the data list is empty or it's the
        # first epoch
        if len(data) > 1:
            poly = np.poly1d(np.polyfit(epochs[-n_backwards:],
                                        data[-n_backwards:], self.poly_degree))
            future_values = [poly(i) for i in future_epochs]
            plot_ax.plot(future_epochs, future_values, linestyle, label=label)
