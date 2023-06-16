import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree

df = pd.read_csv("data/iris.csv")
df_train, df_test = train_test_split(df, test_size=0.3, random_state=13)


def accuracies(alpha):
    dt = DecisionTree(df_train, max_height=25, ccp_alpha=alpha)
    dt.fit()
    height = dt.prune()

    train_acc, test_acc = dt.accuracy(df_train), dt.accuracy(df_test)
    print(
        f"alpha={str(round(alpha, 5)).ljust(5 + 2, '0')} height={str(height).zfill(2)} test={round(test_acc, 2)}%"
    )

    return train_acc, test_acc


def plot(xs, ys_train, ys_test):
    _, ax = plt.subplots()

    ax.plot(xs, ys_train, label="Train", marker="o", drawstyle="steps-post")
    ax.plot(xs, ys_test, label="Test", marker="o", drawstyle="steps-post")
    plt.yticks(np.arange(70, 100 + 5, 5))

    ax.set_xlabel("Cost-Complexity Pruning Alpha")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs CCP Alpha")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    xs = np.linspace(0, 0.05, 10)

    pool = multiprocessing.Pool(8)
    results = pool.map(accuracies, xs)
    pool.close()
    pool.join()

    plot(xs, *zip(*results))
