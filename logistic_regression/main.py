import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
import os


def load_dataset():

    dataset_train = h5py.File("./data/train_catvnoncat.h5", "r")
    x_train_orig = np.array(dataset_train["train_set_x"])
    x_train_orig = x_train_orig / 255.
    y_train = np.array(dataset_train["train_set_y"])
    y_train = np.reshape(y_train, (1, len(y_train)))

    dataset_test = h5py.File("./data/test_catvnoncat.h5", "r")
    x_test_orig = np.array(dataset_test["test_set_x"])
    x_test_orig = x_test_orig / 255.
    y_test = np.array(dataset_test["test_set_y"])
    y_test = np.reshape(y_test, (1, len(y_test)))

    return x_train_orig, y_train, x_test_orig, y_test


def unroll(data):

    return np.reshape(data, (data.shape[0], -1)).T


def initialize_parameters(dim):

    w = np.zeros((1, dim))
    b = 0.0

    parameters = {"w": w,
                  "b": b}

    return parameters


def sigmoid(x):

    return np.divide(1, 1 + np.exp(-x))


def forward_propagation(params, x):

    w = params["w"]
    b = params["b"]

    z = np.dot(w, x) + b
    a = sigmoid(z)

    return a


def compute_cost(Y, A):

    m = Y.shape[1]

    return (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))


def backward_propagation(A, X, Y):

    m = A.shape[1]

    dZ = A - Y

    dw = (1 / m) * np.dot(dZ, X.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    grads = {"dw": dw,
             "db": db}

    return grads


def update_parameters(params, grads, learning_rate):

    parameters = copy.deepcopy(params)
    grads = copy.deepcopy(grads)
    w, b = parameters["w"], parameters["b"]
    dw, db = grads["dw"], grads["db"]

    w = w - learning_rate * dw
    b = b - learning_rate * db

    out = {"w": w,
           "b": b}

    return out


def plot_cost(cost_train, cost_test):

    os.makedirs("./figures", exist_ok=True)

    plt.figure(dpi=300, figsize=(10, 6))
    plt.plot(range(len(cost_train)), cost_train, c="b", label="training set")
    plt.plot(range(len(cost_test)), cost_test, c="r", label="test set")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig("./figures/cost_function.png")


def plot_accuracy(acc_train, acc_test):

    os.makedirs("./figures", exist_ok=True)

    plt.figure(dpi=300, figsize=(10, 6))
    plt.plot(range(len(acc_train)), acc_train, c="b", label="training set")
    plt.plot(range(len(acc_test)), acc_test, c="r", label="test set")
    plt.legend()
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.savefig("./figures/accuracy.png")


def cal_accuracy(yhat, y):

    accuracy = yhat[0, :] == y[0, :]
    accuracy = accuracy.astype(int)

    return sum(accuracy) / y.shape[1]


def prediction(a):

    out = np.zeros_like(a)
    out[0, :] = a[0, :] >= 0.5
    return out


def main():
    # arrays to hold the best costs and accuracies per each iteration
    costs_train = []
    costs_test = []
    accuracy_train = []
    accuracy_test = []

    # load dataset
    x_train, y_train, x_test, y_test = load_dataset()

    # unroll matrix of features
    x_train = unroll(x_train)
    x_test = unroll(x_test)

    # extract number of training set, test set and number of features
    num_features = x_train.shape[0]

    # number of iteration and learning rate
    max_iteration = 10000
    learning_rate = 0.001

    # initialize parameters with zeros
    parameters = initialize_parameters(num_features)

    for iter_main in range(max_iteration):

        a_train = forward_propagation(parameters, x_train)

        cost_train = compute_cost(y_train, a_train)

        costs_train.append(cost_train)

        y_pred_train = prediction(a_train)

        train_acc = cal_accuracy(y_pred_train, y_train)

        accuracy_train.append(train_acc)

        a_test = forward_propagation(parameters, x_test)

        cost_test = compute_cost(y_test, a_test)

        costs_test.append(cost_test)

        y_pred_test = prediction(a_test)

        test_acc = cal_accuracy(y_pred_test, y_test)

        accuracy_test.append(test_acc)

        grads = backward_propagation(a_train, x_train, y_train)

        parameters = update_parameters(parameters, grads, learning_rate)


    plot_cost(costs_train, costs_test)
    plot_accuracy(accuracy_train, accuracy_test)

    print("training accuracy: ", accuracy_train[-1])
    print("test accuracy: ", accuracy_test[-1])


if __name__ == "__main__":

    main()
