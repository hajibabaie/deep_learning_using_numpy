import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
import os


def load_dataset():

    dataset_train = h5py.File("./data/train_catvnoncat.h5", "r")
    train_x_orig = np.array(dataset_train["train_set_x"])
    train_x_orig = train_x_orig / 255.
    train_y_orig = np.array(dataset_train["train_set_y"])
    train_y_orig = np.reshape(train_y_orig, (1, len(train_y_orig)))

    dataset_test = h5py.File("./data/test_catvnoncat.h5", "r")
    test_x_orig = np.array(dataset_test["test_set_x"])
    test_x_orig = test_x_orig / 255.
    test_y_orig = np.array(dataset_test["test_set_y"])
    test_y_orig = np.reshape(test_y_orig, (1, len(test_y_orig)))

    return train_x_orig, train_y_orig, test_x_orig, test_y_orig


def unroll(x):

    return np.reshape(x, (x.shape[0], -1)).T


def initialize_parameters(dims):

    out = {}

    for i in range(1, len(dims)):

        out["W" + str(i)] = np.random.randn(dims[i], dims[i - 1]) * 0.01
        out["b" + str(i)] = np.zeros((dims[i], 1))

    return out


def sigmoid(z):

    return np.divide(1, 1 + np.exp(-z))


def relu(z):

    return np.maximum(z, 0)


def forward_propagation(x, params):

    caches = {"A0": x}
    for i in range(1, len(params) // 2):

        caches["Z" + str(i)] = np.dot(params["W" + str(i)], caches["A" + str(i - 1)]) + params["b" + str(i)]
        caches["A" + str(i)] = sigmoid(caches["Z" + str(i)])

    caches["Z" + str(i + 1)] = np.dot(params["W" + str(i + 1)], caches["A" + str(i)]) + params["b" + str(i + 1)]
    AL = sigmoid(caches["Z" + str(i + 1)])

    return AL, caches


def compute_cost(A, Y):

    m = Y.shape[1]

    cost = np.squeeze((-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A))))

    return cost


def prediction(A):

    y_pred = np.zeros_like(A)

    y_pred[0, :] = A[0, : ] >= 0.5

    return y_pred


def accuracy(y_hat, y):

    return sum(y_hat[0, :] == y[0, :]) / y.shape[1]


def plot(train, test, name):

    os.makedirs("./figures", exist_ok=True)

    plt.figure(dpi=300, figsize=(10, 6))
    plt.plot(range(len(train)), train, label="training set")
    plt.plot(range(len(test)), test, label="test set")
    plt.xlabel("Number of Iteration")
    plt.ylabel(f"{name}")
    plt.legend()
    plt.savefig(f"./figures/{name}.png")


def backward_propagation(AL, Y, parameters, caches, learning_rate):
    m = AL.shape[1]
    number_of_layers = len(caches) // 2
    grads = {}
    grads["dA" + str(number_of_layers)] = np.divide(-Y, AL) + np.divide(1 - Y, 1 - AL)
    for i in reversed(range(1, number_of_layers + 1)):

        if i == number_of_layers:

            grads["dZ" + str(i)] = copy.deepcopy(grads["dA" + str(i)] * AL * (1 - AL))

        else:

            grads["dZ" + str(i)] = grads["dA" + str(i)] * caches["A" + str(i)] * (1 - caches["A" + str(i)])

        grads["dW" + str(i)] = (1 / m) * np.dot(grads["dZ" + str(i)], caches["A" + str(i - 1)].T)
        grads["db" + str(i)] = (1 / m) * np.sum(grads["dZ" + str(i)], axis=1, keepdims=True)
        grads["dA" + str(i - 1)] = np.dot(parameters["W" + str(i)].T, grads["dZ" + str(i)])

    return grads


def update_parameters(grads, parameters, learning_rate):

    for i in range(1, len(parameters) // 2 + 1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * grads["db" + str(i)]

    return parameters


def main():

    train_costs = []
    test_costs = []
    train_accuracy = []
    test_accuracy = []

    # load dataset
    x_train, y_train, x_test, y_test = load_dataset()

    # unroll dataset
    x_train = unroll(x_train)
    x_test = unroll(x_test)

    # number of units per each layer
    layer_dims = [x_train.shape[0], 10, y_train.shape[0]]
    max_iteration = 10000
    learning_rate = 0.0075

    # initialize parameters
    parameters = initialize_parameters(layer_dims)

    for iter_main in range(max_iteration):

        AL_train, caches = forward_propagation(x_train, parameters)

        train_cost = compute_cost(AL_train, y_train)

        train_costs.append(train_cost)

        y_pred_train = prediction(AL_train)

        train_acc = accuracy(y_pred_train, y_train)

        train_accuracy.append(train_acc)

        AL_test, _ = forward_propagation(x_test, parameters)

        test_cost = compute_cost(AL_test, y_test)

        test_costs.append(test_cost)

        y_pred_test = prediction(AL_test)

        test_acc = accuracy(y_pred_test, y_test)

        test_accuracy.append(test_acc)

        grads = backward_propagation(AL_train, y_train, parameters, caches, learning_rate)

        parameters = update_parameters(grads, parameters, learning_rate)

        if iter_main % 100 == 0:
            print("iteration: {}, training cost: {:0.3f}, test cost: {:0.3f}, training acc: {:0.3f}, test acc: {:0.3f}"
                  .format(iter_main, train_cost, test_cost, train_acc, test_acc))

    plot(train_costs, test_costs, "cost")
    plot(train_accuracy, test_accuracy, "accuracy")


if __name__ == "__main__":

    main()

