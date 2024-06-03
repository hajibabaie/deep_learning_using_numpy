import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


class NeuralNetworks:


    def __init__(self, train_name, test_name, units_per_hidden_layers, learning_rate, epochs):

        self._train_name = train_name
        self._test_name = test_name
        self._units_h_layers = units_per_hidden_layers
        self._L = len(self._units_h_layers) + 2
        self._a = {"train": {}, "test": {}}
        self._da = {}
        self._y = {"train": None, "test": None}
        self._n = {i: self._units_h_layers[i - 1] for i in range(1, self._L - 1)}
        self._m = {"train": None, "test": None}
        self._w = {}
        self._dw = {}
        self._b = {}
        self._db = {}
        self._z = {"train": {}, "test": {}}
        self._dz = {}
        self._costs = {"train": [], "test": []}
        self._acc = {"train": [], "test": []}
        self._lr = learning_rate
        self._max_epochs = epochs



    def _load_dataset(self):

        dataset_train = h5py.File(f"./data/{self._train_name}", "r")
        train_x = np.array(dataset_train["train_set_x"]) / 255
        self._a["train"][0] = np.reshape(train_x, (train_x.shape[0], -1)).T
        self._n[0], self._m["train"] = self._a["train"][0].shape

        train_y = np.array(dataset_train["train_set_y"])
        self._y["train"] = np.reshape(train_y, (1, len(train_y)))
        self._n[self._L - 1] = self._y["train"].shape[0]

        dataset_test = h5py.File(f"./data/{self._test_name}", "r")
        test_x = np.array(dataset_test["test_set_x"]) / 255.
        self._a["test"][0] = np.reshape(test_x, (test_x.shape[0], -1)).T
        self._m["test"] = self._a["test"][0].shape[1]

        test_y = np.array(dataset_test["test_set_y"])
        self._y["test"] = np.reshape(test_y, (1, len(test_y)))

    def _initialize_parameters(self):

        for i in range(1, self._L):

            self._w[i] = np.random.randn(self._n[i], self._n[i - 1]) * 0.01
            self._b[i] = np.zeros((self._n[i], 1))

    @staticmethod
    def _sigmoid(i):

        return np.divide(1, 1 + np.exp(-i))

    @staticmethod
    def _tanh(i):

        return np.divide(np.exp(i) - np.exp(-i), np.exp(i) + np.exp(-i))


    def _forward_prop(self, name):

        for i in range(1, self._L):

            self._z[name][i] = np.dot(self._w[i], self._a[name][i - 1]) + self._b[i]

            if i != self._L - 1:

                self._a[name][i] = self._tanh(self._z[name][i])

            else:

                self._a[name][i] = self._sigmoid(self._z[name][i])

    def _compute_cost(self, name):

        return (-1 / self._m[name]) * np.sum(np.multiply(self._y[name], np.log(self._a[name][self._L - 1])) +
                                             np.multiply(1 - self._y[name], np.log(1 - self._a[name][self._L - 1])))

    def _compute_acc(self, name):

        preds = self._a[name][self._L - 1] >= 0.5

        preds = preds.astype(int)

        return np.mean(preds == self._y[name])

    def _backward_prop(self):

        self._da[self._L - 1] = np.divide(-self._y["train"], self._a["train"][self._L - 1]) + np.divide(1 - self._y["train"], 1 - self._a["train"][self._L - 1])

        self._dz[self._L - 1] = self._a["train"][self._L - 1] - self._y["train"]


        for i in reversed(range(1, self._L)):


            self._dw[i] = (1 / self._m["train"]) * np.dot(self._dz[i], self._a["train"][i - 1].T)

            self._db[i] = (1 / self._m["train"]) * np.sum(self._dz[i], axis=1, keepdims=True)

            self._da[i - 1] = np.dot(self._w[i].T, self._dz[i])

            self._dz[i - 1] = np.multiply(self._da[i - 1], 1 - np.square(self._a["train"][i - 1]))

    def _update_parameters(self):

        for i in range(1, self._L):

            self._w[i] = self._w[i] - self._lr * self._dw[i]
            self._b[i] = self._b[i] - self._lr * self._db[i]


    def _plot(self):

        os.makedirs("./figures", exist_ok=True)

        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(self._costs["train"], color="red", label="train")
        plt.plot(self._costs["test"], color="blue", label="test")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Costs")
        plt.legend()
        plt.savefig("./figures/costs.png")

        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(self._acc["train"], color="red", label="train")
        plt.plot(self._acc["test"], color="blue", label="test")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("./figures/accuracy.png")


    def solve(self):

        self._load_dataset()

        self._initialize_parameters()

        for iter_main in range(self._max_epochs):

            self._forward_prop("train")

            self._costs["train"].append(self._compute_cost("train"))

            self._acc["train"].append(self._compute_acc("train"))

            self._forward_prop("test")

            self._costs["test"].append(self._compute_cost("test"))

            self._acc["test"].append(self._compute_acc("test"))

            self._backward_prop()

            self._update_parameters()

            if iter_main % 100 == 0 or iter_main == self._max_epochs - 1:
                print(f"epoch: {iter_main}, train_cost: {self._costs['train'][-1]} train_acc: {self._acc['train'][-1]} test_cost: {self._costs['test'][-1]} test_acc: {self._acc['test'][-1]}")

        self._plot()

        return self._a, self._y, self._n, self._m, self._w, self._b, self._z, self._costs, self._acc, self._da, self._dz, self._dw, self._db


if __name__ == "__main__":

    neural_network = NeuralNetworks("train_catvnoncat.h5", "test_catvnoncat.h5", [32, 16], 0.01, 10000)

    a, y, n, m, w, b, z, costs, accuracies, da, dz, dw, db = neural_network.solve()
