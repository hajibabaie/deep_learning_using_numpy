import os

import h5py
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, train_name, test_name, learning_rate, max_iteration):

        self._train_name = train_name
        self._test_name = test_name
        self._lr = learning_rate
        self._max_iteration = max_iteration
        self._m = {"train": None, "test": None}
        self._n = None
        self._x = {"train": None, "test": None}
        self._y = {"train": None, "test": None}
        self._w = None
        self._dw = None
        self._b = None
        self._db = None
        self._Z = {"train": None, "test": None}
        self._dZ = None
        self._A = {"train": None, "test": None}
        self._dA = None
        self._costs = {"train": [], "test": []}
        self._accuracies = {"train": [], "test": []}


    def _load_data(self):

        train_set = h5py.File(f".\\Data\\{self._train_name}.h5", "r")
        train_set_x = np.array(train_set["train_set_x"]) / 255.

        self._m["train"] = train_set_x.shape[0]
        self._x["train"] = np.reshape(train_set_x, (self._m["train"], -1)).T
        self._n = self._x["train"].shape[0]

        train_set_y = np.array(train_set["train_set_y"])
        self._y["train"] = np.reshape(train_set_y, (1, self._m["train"]))

        test_set = h5py.File(f".\\Data\\{self._test_name}.h5", "r")
        test_set_x = np.array(test_set["test_set_x"]) / 255.

        self._m["test"] = test_set_x.shape[0]
        self._x["test"] = np.reshape(test_set_x, (self._m["test"], -1)).T

        test_set_y = np.array(test_set["test_set_y"])
        self._y["test"] = np.reshape(test_set_y, (1, self._m["test"]))

    @staticmethod
    def _sigmoid(z):

        return np.divide(1, 1 + np.exp(-z))

    def _initialization(self):

        self._w = np.zeros((1, self._n))
        self._b = 0

    def _forward_prop(self, name):

        self._Z[name] = np.dot(self._w, self._x[name]) + self._b # (1, n) * (n, m_train)
        self._A[name] = self._sigmoid(self._Z[name])

    def _compute_cost(self, name):

        cost = (-1 / self._m[name]) * np.sum(np.multiply(self._y[name], np.log(self._A[name])) +
                                             np.multiply(1 - self._y[name], np.log(1 - self._A[name])))

        return cost

    def _compute_accuracies(self, name):

        out = self._A[name] >= 0.5
        acc = out == self._y[name].ravel()

        return np.mean(acc)

    def _backward_prop(self):

        # self._dA = np.divide(-self._y["train"], self._A["train"]) + np.divide(1 - self._y["train"], 1 - self._A["train"])

        self._dZ = self._A["train"] - self._y["train"]

        self._dw = (1 / self._m["train"]) * np.dot(self._dZ, self._x["train"].T)

        self._db = (1 / self._m["train"]) * np.sum(self._dZ)

    def _update_parameters(self):

        self._w -= self._lr * self._dw
        self._b -= self._lr * self._db

    def _plot(self):

        os.makedirs(".\\figures", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self._costs["train"], label="train", color=(1, 0, 0))
        plt.plot(self._costs["test"], label="test", color=(0, 0, 1))
        plt.ylabel("cost")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self._accuracies["train"], label="train", color=(1, 0, 0))
        plt.plot(self._accuracies["test"], label="test", color=(0, 0, 1))
        plt.ylabel("accuracy")
        plt.savefig(".\\figures\cost_accuracy.png")

    def run(self):

        self._load_data()

        self._initialization()

        self._initialization()

        for iter_main in range(self._max_iteration):

            self._forward_prop("train")

            self._costs["train"].append(self._compute_cost("train"))

            self._accuracies["train"].append(self._compute_accuracies("train"))

            self._forward_prop("test")

            self._costs["test"].append(self._compute_cost("test"))

            self._accuracies["test"].append(self._compute_accuracies("test"))

            self._backward_prop()

            self._update_parameters()

            if iter_main % 100 == 0 or iter_main == self._max_iteration - 1:
                print(f"iteration: {iter_main}, train_cost: {self._costs['train'][-1]}, "
                      f"train_accuracy: {self._accuracies['train'][-1]}, test_cost: {self._costs['test'][-1]}, "
                      f"test_accuracy: {self._accuracies['test'][-1]}")

        self._plot()

        out = {
            "x": self._x,
            "y": self._y,
            "w": self._w,
            "b": self._b,
            "dw": self._dw,
            "db": self._db,
            "Z": self._Z,
            "dZ": self._dZ,
            "A": self._A,
            "dA": self._dA,
            "cost": self._costs,
            "accuracy": self._accuracies

        }

        return out
