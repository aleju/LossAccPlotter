from __future__ import print_function, division
from laplotter import LossAccPlotter
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("------------------")
    print("Chart 1")
    print("150 datapoints")
    print("x-index 5 of loss_train should create a warning as its set to NaN")
    print("------------------")
    nb_points = 150

    # example values for loss and accuracy
    loss_train = np.linspace(0.8, 0.1, num=nb_points)
    loss_val = np.linspace(0.7, 0.2, num=nb_points)
    acc_train = np.linspace(0.5, 0.85, num=nb_points)
    acc_val = np.linspace(0.6, 0.75, num=nb_points)

    # add noise
    loss_train = loss_train + np.random.normal(scale=0.05, size=nb_points)
    loss_val = loss_val + np.random.normal(scale=0.1, size=nb_points)
    acc_train = acc_train + np.random.normal(scale=0.05, size=nb_points)
    acc_val = acc_val + np.random.normal(scale=0.1, size=nb_points)

    loss_train[5] = float("nan")

    lap = LossAccPlotter()
    for idx in range(nb_points):
        lap.add_values(idx,
                       loss_train=loss_train[idx], loss_val=loss_val[idx],
                       acc_train=acc_train[idx], acc_val=acc_val[idx],
                       redraw=False)
    lap._redraw()
    
    print("Close the chart to continue.")
    lap.block()

    

if __name__ == "__main__":
    main()
