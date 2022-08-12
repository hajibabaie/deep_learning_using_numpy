import os

import matplotlib.pyplot as plt
import numpy as np
import h5py


def load_dataset():

    dataset_train = h5py.File("./data/train_catvnoncat.h5", "r")

    train_x_orig = np.array(dataset_train["train_set_x"])
    train_x_orig = train_x_orig / 255.

    train_y_orig = np.array(dataset_train["train_set_y"])
    train_y_orig = np.reshape(train_y_orig, (1, len(train_y_orig)))

    dataset_test = h5py.File("./data/test_catvnoncat.h5")

    test_x_orig = np.array(dataset_test["test_set_x"])
    test_x_orig = test_x_orig / 255.

    test_y_orig = np.array(dataset_test["test_set_y"])
    test_y_orig = np.reshape(test_y_orig, (1, len(test_y_orig)))

    return train_x_orig, train_y_orig, test_x_orig, test_y_orig


def unroll(x):

    out = np.reshape(x, (x.shape[0], -1)).T

    return out


def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    b = 0.0

    return w, b


def sigmoid(x):

    return np.divide(1, 1 + np.exp(-x))


def prediction(a):

    y_prediction = np.zeros_like(a)
    y_prediction[0, :] = a >= 0.5

    return y_prediction


def accuracy(y_pred, y):

    return sum(y_pred[0, :] == y[0, :]) / y.shape[1]


def main():

    best_cost_train = []
    best_cost_test = []

    train_accuracy = []
    test_accuracy = []

    max_iteration = 10000
    learning_rate = 0.001

    # import dataset
    train_x, train_y, test_x, test_y = load_dataset()

    # unroll training and test dataset
    train_x = unroll(train_x)
    test_x = unroll(test_x)

    m_train = train_x.shape[1]
    m_test = test_x.shape[1]
    num_features = train_x.shape[0]

    # initialize parameters
    w, b = initialize_with_zeros(train_x.shape[0])

    for iter_main in range(max_iteration):

        z = np.dot(w.T, train_x) + b
        a = sigmoid(z)
        cost = (-1 / m_train) * np.sum(np.multiply(train_y, np.log(a)) + np.multiply(1 - train_y, np.log(1 - a)))
        best_cost_train.append(cost)

        train_pred = prediction(a)
        train_acc = accuracy(train_pred, train_y)
        train_accuracy.append(train_acc)


        z_test = np.dot(w.T, test_x) + b
        a_test = sigmoid(z_test)
        cost_test = (-1 / m_test) * np.sum(np.multiply(test_y, np.log(a_test)) + np.multiply(1 - test_y, np.log(1 - a_test)))
        best_cost_test.append(cost_test)

        test_pred = prediction(a_test)
        test_acc = accuracy(test_pred, test_y)
        test_accuracy.append(test_acc)

        dw = (1 / m_train) * np.dot(train_x, (a - train_y).T)
        db = (1 / m_train) * np.sum(a - train_y)
        w = w - learning_rate * dw
        b = b - learning_rate * db




    os.makedirs("./figures", exist_ok=True)
    plt.figure(dpi=300)
    plt.plot(range(max_iteration), best_cost_train, c="red", label="training set")
    plt.plot(range(max_iteration), best_cost_test, c="blue", label="test set")
    plt.legend()
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.savefig("./figures/cost_function.png")

    plt.figure(dpi=300)
    plt.plot(range(max_iteration), train_accuracy, c="red", label="training set")
    plt.plot(range(max_iteration), test_accuracy, c="blue", label="test set")
    plt.legend()
    plt.xlabel("Number of Iteration")
    plt.ylabel("Accuracy")
    plt.savefig("./figures/accuracy.png")




    print(train_accuracy[-1])
    print(test_accuracy[-1])



if __name__ == "__main__":

    main()
