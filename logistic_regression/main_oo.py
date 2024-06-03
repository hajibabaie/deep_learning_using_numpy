import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


class LogisticRegression:

    def __init__(self, name_train_set, name_test_set, learning_rate, number_of_iterations):

        self._name_train = name_train_set
        self._name_test = name_test_set
        self._lr = learning_rate
        self._num_iterations = number_of_iterations
        self._x = {"train": None, "test": None}
        self._y = {"train": None, "test": None}
        self._Z = {"train": None, "test": None}
        self._m = {"train": None, "test": None}
        self._n = None
        self._A = {"train": [], "test": []}
        self._dZ = None
        self._w = None
        self._dw = None
        self._b = None
        self._db = None
        self._costs = {"train": [], "test": []}
        self._accuracy = {"train": [], "test": []}

    def _load_data(self):

        dataset_train = h5py.File(f"./data/{self._name_train}.h5", "r")
        x_train_orig = np.array(dataset_train["train_set_x"]) / 255.
        self._x["train"] = np.reshape(x_train_orig, (x_train_orig.shape[0], -1)).T
        self._n, self._m["train"] = self._x["train"].shape

        y_train = np.array(dataset_train["train_set_y"])
        self._y["train"] = np.reshape(y_train, (1, len(y_train)))

        dataset_test = h5py.File(f"./data/{self._name_test}.h5", "r")
        x_test_orig = np.array(dataset_test["test_set_x"]) / 255.
        self._x["test"] = np.reshape(x_test_orig, (x_test_orig.shape[0], -1)).T
        self._m["test"] = self._x["test"].shape[1]

        y_test = np.array(dataset_test["test_set_y"])
        self._y["test"] = np.reshape(y_test, (1, len(y_test)))


    def _initialize_parameters(self):

        self._w = np.zeros((1, self._n))
        self._b = 0.

    @staticmethod
    def _sigmoid(x):

        return np.divide(1, 1 + np.exp(-x))

    def _forward_prop(self, name):

        self._Z[f"{name}"] = np.dot(self._w, self._x[f"{name}"]) + self._b
        self._A[f"{name}"] = self._sigmoid(self._Z[f"{name}"])

    def _compute_cost(self, name):

        cost = (-1 / self._m[f"{name}"]) * np.sum(np.multiply(self._y[f"{name}"], np.log(self._A[f"{name}"])) +
                                                  np.multiply(1 - self._y[f"{name}"], np.log(1 - self._A[f"{name}"])))

        return cost

    def _backward_prop(self):

        self._dZ = self._A["train"] - self._y["train"]

        self._dw = (1 / self._m["train"]) * np.dot(self._dZ, self._x["train"].T)
        self._db = (1 / self._m["train"]) * np.sum(self._dZ)

    def _update_parameters(self):

        self._w -= self._lr * self._dw
        self._b -= self._lr * self._db

    def _calc_accuracy(self, name):

        out = np.zeros_like(self._A[f"{name}"])
        out[0, :] = self._A[f"{name}"][0, :] >= 0.5

        acc = np.sum(out[0, :].astype(int) == self._y[f"{name}"]) / self._m[f"{name}"]
        self._accuracy[f"{name}"].append(acc)

    def _plot(self):

        os.makedirs("./figures_oo", exist_ok=True)

        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(self._costs["train"], color="blue", label="train")
        plt.plot(self._costs["test"], color="red", label="test")
        plt.xlabel("Number of Iteration")
        plt.ylabel("costs")
        plt.savefig(f"./figures_oo/costs.png")

        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(self._accuracy["train"], color="blue", label="train")
        plt.plot(self._accuracy["test"], color="red", label="test")
        plt.xlabel("Number of Iteration")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(f"./figures_oo/accuracy.png")


    def solve(self):

        self._load_data()

        self._initialize_parameters()

        for iter_main in range(self._num_iterations):

            self._forward_prop("train")

            self._costs["train"].append(self._compute_cost("train"))

            self._calc_accuracy("train")

            self._forward_prop("test")

            self._costs["test"].append(self._compute_cost("test"))

            self._calc_accuracy("test")

            self._backward_prop()

            self._update_parameters()

            if iter_main % 100 == 0 or iter_main == self._num_iterations - 1:
                print(f"iteration: {iter_main}, train_cost: {self._costs['train'][-1]}, "
                      f"train_accuracy: {self._accuracy['train'][-1]}, test_cost: {self._costs['test'][-1]}, "
                      f"test_accuracy: {self._accuracy['test'][-1]}")

        self._plot()

        out = {
            "x": self._x,
            "y": self._y,
            "w": self._w,
            "b": self._b,
            "Z": self._Z,
            "A": self._A,
            "costs": self._costs,
            "accuracy": self._accuracy
        }

        return out



if __name__ == "__main__":

    lr = LogisticRegression("train_catvnoncat", "test_catvnoncat", 0.001, 11000)
    outputs = lr.solve()
