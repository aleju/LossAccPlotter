"""Various checks to make sure that the LossAccPlotter() works decently.
These checks do not run fully automatically, they have to be validated by
a human.

How to run them:
    python check_laplotter.py
"""
from __future__ import print_function, division
from laplotter import LossAccPlotter
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("")
    print("------------------")
    print("1 datapoint")
    print("------------------")
    # generate example values for: loss train, loss validation, accuracy train
    # and accuracy validation
    (loss_train, loss_val, acc_train, acc_val) = create_values(1)

    # generate a plot showing the example values
    show_chart(loss_train, loss_val, acc_train, acc_val,
               title="A single datapoint")

    print("")
    print("------------------")
    print("150 datapoints")
    print("Saved to file 'plot.png'")
    print("------------------")
    (loss_train, loss_val, acc_train, acc_val) = create_values(150)
    show_chart(loss_train, loss_val, acc_train, acc_val,
               lap=LossAccPlotter(save_to_filepath="plot.png"),
               title="150 datapoints, saved to file 'plot.png'")

    print("")
    print("------------------")
    print("150 datapoints")
    print("No accuracy chart")
    print("------------------")
    (loss_train, loss_val, _, _) = create_values(150)
    show_chart(loss_train, loss_val, np.array([]), np.array([]),
               lap=LossAccPlotter(show_acc_plot=False),
               title="150 datapoints, no accuracy chart")

    print("")
    print("------------------")
    print("150 datapoints")
    print("No loss chart")
    print("------------------")
    (_, _, acc_train, acc_val) = create_values(150)
    show_chart(np.array([]), np.array([]), acc_train, acc_val,
               lap=LossAccPlotter(show_loss_plot=False),
               title="150 datapoints, no loss chart")

    print("")
    print("------------------")
    print("150 datapoints")
    print("Only validation values (no training lines)")
    print("------------------")
    (_, loss_val, _, acc_val) = create_values(150)
    show_chart(np.array([]), loss_val, np.array([]), acc_val,
               title="150 datapoints, only validation (no training)")

    print("")
    print("------------------")
    print("150 datapoints")
    print("No regressions")
    print("------------------")
    (loss_train, loss_val, acc_train, acc_val) = create_values(150)
    show_chart(loss_train, loss_val, acc_train, acc_val,
               lap=LossAccPlotter(show_regressions=False),
               title="150 datapoints, regressions deactivated")
    
    print("")
    print("------------------")
    print("150 datapoints")
    print("No averages")
    print("------------------")
    (loss_train, loss_val, acc_train, acc_val) = create_values(150)
    show_chart(loss_train, loss_val, acc_train, acc_val,
               lap=LossAccPlotter(show_averages=False),
               title="150 datapoints, averages deactivated")

    print("")
    print("------------------")
    print("150 datapoints")
    print("x-index 5 of loss_train should create a warning as its set to NaN")
    print("------------------")
    (loss_train, loss_val, acc_train, acc_val) = create_values(150)

    loss_train[5] = float("nan")
    
    show_chart(loss_train, loss_val, acc_train, acc_val,
               title="150 datapoints, one having value NaN (loss train at x=5)")

    print("")
    print("------------------")
    print("1000 datapoints training")
    print("100 datapoints validation")
    print("------------------")
    nb_points_train = 1000
    nb_points_val = 100
    (loss_train, loss_val, acc_train, acc_val) = create_values(nb_points_train)

    # set 9 out of 10 values of the validation arrays to -1.0 (Which will be
    # interpreted as None in show_chart(). Numpy doesnt support None directly,
    # only NaN, which is already used before to check whether the Plotter
    # correctly creates a warning if any data point is NaN.)
    all_indices = np.arange(0, nb_points_train-1, 1)
    keep_indices = np.arange(0, nb_points_train-1, int(nb_points_train / nb_points_val))
    set_to_none_indices = np.delete(all_indices, keep_indices)
    loss_val[set_to_none_indices] = -1.0
    acc_val[set_to_none_indices] = -1.0

    show_chart(loss_train, loss_val, acc_train, acc_val,
               title="1000 training datapoints, but only 100 validation datapoints")

def create_values(nb_points):
    """Generate example (y-)values for all lines with some added random noise.
    
    Args:
        nb_points: Number of example values
    Returns:
        4 numpy arrays of values as a tuple: (array, array, array, array)
        First Array: loss train
        Second Array: loss validation
        Third Array: accuracy train
        Fourth Array: accuracy validation
    """
    # we add a bit more noise (0.1) to the validation data compared to the
    # training data (0.05), which seems more realistic
    lt = add_noise(np.linspace(0.8, 0.1, num=nb_points), 0.05)
    lv = add_noise(np.linspace(0.7, 0.2, num=nb_points), 0.1)
    at = add_noise(np.linspace(0.5, 0.85, num=nb_points), 0.05)
    av = add_noise(np.linspace(0.6, 0.75, num=nb_points), 0.1)
    return (lt, lv, at, av)

def add_noise(values, scale):
    """Add normal distributed noise to an array.
    Args:
        values: Numpy array of values, e.g. [0.7, 0.6, ...]
        scale: Standard deviation of the normal distribution.
    Returns:
        Input array with added noise.
    """
    return values + np.random.normal(scale=scale, size=values.shape[0])

def show_chart(loss_train, loss_val, acc_train, acc_val, lap=None, title=None):
    """Shows a plot using the LossAccPlotter and all provided values.

    Args:
        loss_train: y-values of the loss function of the training dataset.
        loss_val: y-values of the loss function of the validation dataset.
        acc_train: y-values of the accuracy of the training dataset.
        acc_val: y-values of the accuracy of the validation dataset.
        lap: A LossAccPlotter-Instance or None. If None then a new LossAccPlotter
            will be instantiated. (Default is None.)
        title: The title to use for the plot, i.e. LossAccPlotter.title .
    """
    lap = LossAccPlotter() if lap is None else lap
    if title is not None:
        lap.title = title

    for idx in range(loss_train.shape[0]):
        lt = loss_train[idx] if loss_train[idx] != -1.0 else None
        lap.add_values(idx, loss_train=lt, redraw=False)
    for idx in range(loss_val.shape[0]):
        lv = loss_val[idx] if loss_val[idx] != -1.0 else None
        lap.add_values(idx, loss_val=lv, redraw=False)
    for idx in range(acc_train.shape[0]):
        at = acc_train[idx] if acc_train[idx] != -1.0 else None
        lap.add_values(idx, acc_train=at, redraw=False)
    for idx in range(acc_val.shape[0]):
        av = acc_val[idx] if acc_val[idx] != -1.0 else None
        lap.add_values(idx, acc_val=av, redraw=False)

    lap.redraw()

    # block at the end so that the plot does not close immediatly.
    print("Close the chart to continue.")
    lap.block()

if __name__ == "__main__":
    main()
