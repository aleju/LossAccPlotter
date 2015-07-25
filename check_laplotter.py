from laplotter import LossAccPlotter
import matplotlib.pyplot as plt

def main():
    lap = LossAccPlotter()
    lap.add_values(1, loss_train=0.50, loss_val=0.60, acc_train=0.75, acc_val=0.70)
    lap.add_values(2, loss_train=0.40, loss_val=0.55, acc_train=0.80, acc_val=0.72)
    lap.add_values(3, loss_train=0.32, loss_val=0.51, acc_train=0.84, acc_val=0.74)
    lap.block()

if __name__ == "__main__":
    main()
