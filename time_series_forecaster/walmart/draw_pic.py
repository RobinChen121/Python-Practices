"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/12/11 16:43
Description: 
    

"""
import matplotlib
import numpy as np

def draw(sales, seq_len, X, real_pred_train, real_pred_eval):
    matplotlib.use("Qt5Agg")  # 或者 "Qt5Agg"，具体取决于环境中装了哪个: conda install pyqt
    import matplotlib.pyplot as plt

    plt.plot(sales, label="real sales")
    train_size = int(X.shape[0] * 0.8)

    plt.plot(
    np.arange(train_size + seq_len),
    np.concatenate((np.array(sales[:seq_len]), real_pred_train)),
    # np.concatenate((np.array(sales[:seq_len]), real_pred.flatten())),
    label = "predicted sales train",
    )

    plt.plot(
        np.arange(len(sales) - len(real_pred_eval), len(sales)),
        np.array(real_pred_eval),
        # np.concatenate((np.array(sales[:seq_len]), real_pred.flatten())),
        label="predicted sales evaluation",
    )
    plt.grid(True)
    plt.legend()
    plt.show()