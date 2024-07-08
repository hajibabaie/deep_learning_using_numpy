import os

import matplotlib.pyplot as plt
import numpy as np
import h5py



class OneHiddenLayer:

    def __init__(self, train_name, test_name, number_of_units_hidden_layer, learning_rate, max_iteration):

        self._train_name = train_name
        self._test_name = test_name
        self._x = {"train": None, "test": None}
        self._y = {"train": None, "test": None}
        self._w = {1: None, 2: None}
        self._dw = {1: None, 2: None}
        self._b = {1: None, 2: None}
        self._db = {1: None, 2: None}
        self._Z = {1: {"train": None, "test": None}, 2: {"train": None, "test": None}}
        self._dZ = {1: None, 2: None}
        self._A = {1: {"train": None, "test": None}, 2: {"train": None, "test": None}}
        self._dA = {1: None, 2: None}
        self._m = {"train": None, "test": None}
        self._n = {0: None, 1: number_of_units_hidden_layer, 2: None}
        self._cost = {"train": [], "test": []}
        self._accuracy = {"train": [], "test": []}
        self._learning_rate = learning_rate
        self._max_iteration = max_iteration

    def _load(self):

        train_set = h5py.File(f".\data\\{self._train_name}.h5", "r")
        train_set_x = np.array(train_set["train_set_x"])
        train_set_y = np.array(train_set["train_set_y"])
        self._m["train"] = train_set_x.shape[0]
        self._x["train"] = np.reshape(train_set_x, (self._m["train"], -1)).T
        self._n[0] = self._x["train"].shape[0]
        self._y["train"] = np.reshape(train_set_y, (1, self._m["train"]))
        self._n[2] = self._y["train"].shape[0]

        test_set = h5py.File(f".\data\\{self._test_name}.h5", "r")
        test_set_x = np.array(test_set["test_set_x"])
        test_set_y = np.array(test_set["test_set_y"])
        self._m["test"] = test_set_x.shape[0]
        self._x["test"] = np.reshape(test_set_x, (self._m["test"], -1)).T
        self._y["test"] = np.reshape(test_set_y, (1, self._m["test"]))

    def _initialization(self):

        self._w[1] = np.random.randn(self._n[1], self._n[0]) * 0.01
        self._b[1] = np.zeros((self._n[1], 1))

        self._w[2] = np.random.randn(self._n[2], self._n[1]) * 0.01
        self._b[2] = np.zeros((self._n[2], 1))

    @staticmethod
    def _sigmoid(z):

        return np.divide(1, 1 + np.exp(-z))

    @staticmethod
    def _tanh(z):

        return np.divide(np.exp(z) - np.exp(-z), np.exp(z) + np.exp(-z))

    def _relu(self, z):

        return np.maximum(z, 0)

    def _forward_prop(self, name):

        self._Z[1][name] = np.dot(self._w[1], self._x[name]) + self._b[1]

        self._A[1][name] = self._relu(self._Z[1][name])

        self._Z[2][name] = np.dot(self._w[2], self._A[1][name]) + self._b[2]
        self._A[2][name] = self._sigmoid(self._Z[2][name])

    def _compute_cost(self, name):

        cost = (-1 / self._m[name]) * np.sum(np.multiply(self._y[name], np.log(self._A[2][name])) +
                                             np.multiply(1 - self._y[name], np.log(1 - self._A[2][name])))

        return cost

    def _compute_accuracy(self, name):

        out = self._A[2][name] >= 0.5

        return np.mean(out == self._y[name])

    def _backward_prop(self):

        self._dA[2] = np.divide(- self._y["train"], self._A[2]["train"]) + np.divide(1 - self._y["train"], 1 - self._A[2]["train"])

        self._dZ[2] = self._A[2]["train"] - self._y["train"]

        self._dw[2] = (1 / self._m["train"]) * np.dot(self._dZ[2], self._A[1]["train"].T)

        self._db[2] = (1 / self._m["train"]) * np.sum(self._dZ[2], axis=1, keepdims=True)

        self._dA[1] = np.dot(self._w[2].T, self._dZ[2])

        # self._dZ[1] = np.multiply(self._dA[1], np.multiply(self._A[1]["train"], 1 - np.power(self._A[1]["train"], 2)))

        self._dZ[1] = np.maximum(self._dA[1], 0)

        self._dw[1] = (1 / self._m["train"]) * np.dot(self._dZ[1], self._x["train"].T)

        self._db[1] = (1 / self._m["train"]) * np.sum(self._dZ[1], axis=1, keepdims=True)

    def _update_parameters(self):

        self._w[2] -= self._learning_rate * self._dw[2]
        self._b[2] -= self._learning_rate * self._db[2]
        self._w[1] -= self._learning_rate * self._dw[1]
        self._b[1] -= self._learning_rate * self._db[1]

    def _plot(self):

        os.makedirs(".\\figures", exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self._cost["train"], label="train", color=(1, 0, 0))
        plt.plot(self._cost["test"], label="test", color=(0, 0, 1))
        plt.ylabel("cost")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self._accuracy["train"], label="train", color=(1, 0, 0))
        plt.plot(self._accuracy["test"], label="test", color=(0, 0, 1))
        plt.savefig("./figures/cost_acc.png")

    def run(self):

        self._load()

        self._initialization()

        for iter_main in range(self._max_iteration):

            self._forward_prop("train")

            self._cost["train"].append(self._compute_cost("train"))

            self._accuracy["train"].append(self._compute_accuracy("train"))

            self._forward_prop("test")

            self._cost["test"].append(self._compute_cost("test"))

            self._accuracy["test"].append(self._compute_accuracy("test"))

            self._backward_prop()

            self._update_parameters()

            if iter_main % 100 == 0 or iter_main == self._max_iteration - 1:

                print(f"iteration: {iter_main}, train_cost: {self._cost['train'][-1]:0.2f} "
                      f"train_acc: {self._accuracy['train'][-1]:0.2f}, test_cost: {self._cost['test'][-1]:0.2f} "
                      f"test_acc: {self._accuracy['test'][-1]:0.2f}")

        self._plot()

        out = {
            "x": self._x,
            "y": self._y,
            "w": self._w,
            "b": self._b,
            "z": self._Z,
            "a": self._A,
            "dz": self._dZ,
            "dw": self._dw,
            "dA": self._dA,
            "db": self._db,
        }

        return out
