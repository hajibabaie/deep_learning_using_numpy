import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


class OneHiddenLayer:


    def __init__(self, train_name, test_name, units_hidden_layer, learning_rate, max_iteration):

        self._x = {"train": None, "test": None}
        self._y = {"train": None, "test": None}
        self._m = {"train": None, "test": None}
        self._n = {0: None, 1: units_hidden_layer, 2: None}
        self._z = {"train": {1: None, 2: None}, "test": {1: None, 2: None}}
        self._dz = {1: None, 2: None}
        self._a = {"train": {1: None, 2: None}, "test": {1: None, 2: None}}
        self._da = {1: None, 2: None}
        self._cost = {"train": [], "test": []}
        self._acc = {"train": [], "test": []}
        self._w = {1: None, 2: None}
        self._b = {1: None, 2: None}
        self._dw = {1: None, 2: None}
        self._db = {1: None, 2: None}
        self._train_name = train_name
        self._test_name = test_name
        self._lr = learning_rate
        self._max_iteration = max_iteration

    def _load_dataset(self):

        dataset_train = h5py.File(f"./data/{self._train_name}", "r")
        x_train = np.array(dataset_train["train_set_x"]) / 255.
        self._x["train"] = np.reshape(x_train, (x_train.shape[0], -1)).T
        self._m["train"] = self._x["train"].shape[1]
        self._n[0] = self._x["train"].shape[0]
        y_train = np.array(dataset_train["train_set_y"])
        self._y["train"] = np.reshape(y_train, (1, len(y_train)))
        self._n[2] = self._y["train"].shape[0]

        dataset_test = h5py.File(f"./data/{self._test_name}", "r")
        x_test = np.array(dataset_test["test_set_x"]) / 255.
        self._x["test"] = np.reshape(x_test, (x_test.shape[0], -1)).T
        self._m["test"] = self._x["test"].shape[1]
        y_test = np.array(dataset_test["test_set_y"])
        self._y["test"] = np.reshape(y_test, (1, len(y_test)))

    @staticmethod
    def _sigmoid(i):

        return np.divide(1, 1 + np.exp(-i))

    @staticmethod
    def _tanh(i):

        return np.divide(np.exp(i) - np.exp(-i), np.exp(i) + np.exp(-i))

    def _initialize_parameters(self):

        self._w[1] = np.random.randn(self._n[1], self._n[0]) * 0.01
        self._b[1] = np.zeros((self._n[1], 1))

        self._w[2] = np.random.randn(self._n[2], self._n[1]) * 0.01
        self._b[2] = np.zeros((self._n[2], 1))

    def _forward_prop(self, name):

        self._z[name][1] = np.dot(self._w[1], self._x[name]) + self._b[1]
        self._a[name][1] = self._tanh(self._z[name][1])

        self._z[name][2] = np.dot(self._w[2], self._a[name][1]) + self._b[2]
        self._a[name][2] = self._sigmoid(self._z[name][2])

    def _compute_cost(self, name):

        return (-1 / self._m[name]) * np.sum(np.multiply(self._y[name], np.log(self._a[name][2])) +
                                             np.multiply(1 - self._y[name], np.log(1 - self._a[name][2])))

    def _compute_accuracy(self, name):

        preds = self._a[name][2] >= 0.5

        preds = preds.astype(int)

        return np.mean(preds == self._y[name])

    def _backward_prop(self):

        self._dz[2] = self._a["train"][2] - self._y["train"]

        self._dw[2] = (1 / self._m["train"]) * np.dot(self._dz[2], self._a["train"][1].T)

        self._db[2] = (1 / self._m["train"]) * np.sum(self._dz[2], axis=1, keepdims=True)

        self._da[1] = np.dot(self._w[2].T, self._dz[2])

        self._dz[1] = np.multiply(self._da[1], 1 - np.square(self._a["train"][1]))

        self._dw[1] = (1 / self._m["train"]) * np.dot(self._dz[1], self._x["train"].T)

        self._db[1] = (1 / self._m["train"]) * np.sum(self._dz[1], axis=1, keepdims=True)

    def _update_parameters(self):

        self._w[2] -= self._lr * self._dw[2]
        self._b[2] -= self._lr * self._db[2]

        self._w[1] -= self._lr * self._dw[1]
        self._b[1] -= self._lr * self._db[1]

    def _plot(self):

        os.makedirs("./figures_oo", exist_ok=True)

        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(self._cost["train"], color="blue", label="train")
        plt.plot(self._cost["test"], color="red", label="test")
        plt.legend()
        plt.savefig("./figures_oo/cost.png")

        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(self._acc["train"], color="blue", label="train")
        plt.plot(self._acc["test"], color="red", label="test")
        plt.legend()
        plt.savefig("./figures_oo/acc.png")

    def solve(self):

        self._load_dataset()

        self._initialize_parameters()

        for iter_main in range(self._max_iteration):

            self._forward_prop("train")

            self._cost["train"].append(self._compute_cost("train"))

            self._acc["train"].append(self._compute_accuracy("train"))

            self._forward_prop("test")

            self._cost["test"].append(self._compute_cost("test"))

            self._acc["test"].append(self._compute_accuracy("test"))

            self._backward_prop()

            self._update_parameters()

            if iter_main % 100 == 0 or iter_main == self._max_iteration - 1:
                print(f"iteration: {iter_main}, train_cost: {self._cost['train'][-1]} "
                      f"train_acc: {self._acc['train'][-1]}, test_cost: {self._cost['test'][-1]} "
                      f"test_acc: {self._acc['test'][-1]}")

        self._plot()

        return self._x, self._y, self._w, self._b, self._z, self._a, self._cost, self._acc, self._da, self._dz, self._dw, self._db



if __name__ == "__main__":

    neural_network = OneHiddenLayer("train_catvnoncat.h5", "test_catvnoncat.h5", 5, 0.001, 15000)

    x, y, w, b, z, a, cost, accuracy, da, dz, dw, db = neural_network.solve()
