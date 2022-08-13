import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
import os


def load_dataset():

    dataset_train = h5py.File("./data/train_catvnoncat.h5", "r")
    x_train = np.array(dataset_train["train_set_x"])
    x_train = x_train / 255.
    y_train = np.array(dataset_train["train_set_y"])
    y_train = np.reshape(y_train, (1, len(y_train)))

    dataset_test = h5py.File("./data/test_catvnoncat.h5", "r")
    x_test = np.array(dataset_test["test_set_x"])
    x_test = x_test / 255.
    y_test = np.array(dataset_test["test_set_y"])
    y_test = np.reshape(y_test, (1, len(y_test)))

    return x_train, y_train, x_test, y_test


def unroll(x):

    return np.reshape(x, (x.shape[0], -1)).T


def initialize_parameters(n_in, n_second, n_out):

    W1 = np.random.randn(n_second, n_in) * 0.01
    b1 = np.zeros((n_second, 1))
    W2 = np.random.randn(n_out, n_second) * 0.01
    b2 = np.zeros((n_out, 1))

    out = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    return out


def tanh(x):

    return np.divide(np.exp(x) - np.exp(-x), np.exp(x) + np.exp(-x))


def sigmoid(x):

    return np.divide(1, 1 + np.exp(-x))


def forward_propagation(params, x):

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = np.dot(W1, x) + b1
    A1 = tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    out = {
        "X": x,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2
    }
    return out, A2


def compute_cost(A2, Y):

    m = Y.shape[1]

    return (-1 / m) * np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2)))


def prediction(a):

    out = np.zeros_like(a)

    out[0, :] = a[0, :] >= 0.5

    return out


def calc_accuracy(yhat, y):

    accuracy = sum(yhat[0, :] == y[0, :]) / y.shape[1]

    return accuracy


def backward_propagation(a2, y, cache, parameters):

    A1 = cache["A1"]
    W2 = parameters["W2"]
    X = cache["X"]
    m = y.shape[1]

    dA2 = np.divide(-y, a2) + np.divide(1 - y, 1 - a2)
    dZ2 = a2 - y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, 1 - np.square(A1))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dA2": dA2,
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
        "dA1": dA1,
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1
    }

    return grads


def update_parameters(params, grads, learning_rate):

    parameters = copy.deepcopy(params)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    out = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    return out


def plot(train, test, name):

    os.makedirs("./figures", exist_ok=True)

    plt.figure(dpi=300, figsize=(10, 6))
    plt.plot(range(len(train)), train, c="b", label="training set")
    plt.plot(range(len(test)), test, c="r", label="test set")
    plt.legend()
    plt.xlabel("Number of Iteration")
    plt.ylabel(f"{name}")
    plt.savefig(f"./figures/{name}.png")


def main():

    # arrays to hold the best costs and accuracies per each iteration
    costs_train = []
    costs_test = []
    accuracy_train = []
    accuracy_test = []

    # load training and test dataset
    x_train, y_train, x_test, y_test = load_dataset()

    # unroll matrices of features
    x_train = unroll(x_train)
    x_test = unroll(x_test)

    # number of units per each layer
    n_0 = x_train.shape[0]
    n_1 = 100
    n_2 = y_train.shape[0]

    # max iterations and learning rate
    max_iteration = 10000
    learning_rate = 0.001

    # initialize parameters
    parameters = initialize_parameters(n_0, n_1, n_2)


    for iter_main in range(max_iteration):
        cache_train, A2_train = forward_propagation(parameters, x_train)

        cost_train = compute_cost(A2_train, y_train)

        costs_train.append(cost_train)

        y_pred_train = prediction(A2_train)

        acc_train = calc_accuracy(y_pred_train, y_train)

        accuracy_train.append(acc_train)

        _, A2_test = forward_propagation(parameters, x_test)

        cost_test = compute_cost(A2_test, y_test)

        costs_test.append(cost_test)

        y_pred_test = prediction(A2_test)

        acc_test = calc_accuracy(y_pred_test, y_test)

        accuracy_test.append(acc_test)

        grads = backward_propagation(A2_train, y_train, cache_train, parameters)

        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
        print(f"{iter_main}: training cost: {cost_train}, test cost: {cost_test}, train acc: {acc_train}, test acc: {acc_test}")


    plot(costs_train, costs_test, "cost")
    plot(accuracy_train, accuracy_test, "accuracy")


if __name__ == "__main__":

    main()


