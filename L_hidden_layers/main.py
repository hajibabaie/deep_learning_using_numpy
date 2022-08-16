import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py


def load_data():

    train_dataset = h5py.File("./datasets/train_signs.h5", "r")
    x_train = tf.data.Dataset.from_tensor_slices(train_dataset["train_set_x"])
    y_train = tf.data.Dataset.from_tensor_slices(train_dataset["train_set_y"])

    test_dataset = h5py.File("./datasets/test_signs.h5", "r")
    x_test = tf.data.Dataset.from_tensor_slices(test_dataset["test_set_x"])
    y_test = tf.data.Dataset.from_tensor_slices(test_dataset["test_set_y"])

    return x_train, y_train, x_test, y_test


def normalize(image):

    image = tf.cast(image, tf.float32) / 255.
    image = tf.reshape(image, (-1,))
    return image


def one_hot_matrix(label):

    one_hot = tf.one_hot(label, depth=6, axis=0)
    one_hot = tf.reshape(one_hot, (-1,))
    return one_hot


def initialize_parameters():

    initializer = tf.keras.initializers.GlorotNormal()

    W1 = tf.Variable(initializer(shape=(25, 12288)))
    b1 = tf.Variable(initializer(shape=(25, 1)))
    W2 = tf.Variable(initializer(shape=(12, 25)))
    b2 = tf.Variable(initializer(shape=(12, 1)))
    W3 = tf.Variable(initializer(shape=(6, 12)))
    b3 = tf.Variable(initializer(shape=(6, 1)))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    return parameters


def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    z1 = tf.add(tf.matmul(W1, X), b1)
    a1 = tf.keras.activations.relu(z1)

    z2 = tf.add(tf.matmul(W2, a1), b2)
    a2 = tf.keras.activations.relu(z2)

    z3 = tf.add(tf.matmul(W3, a2), b3)

    return z3


def compute_cost(labels, logits):

    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True))


def plot(train, test, name):

    os.makedirs("./figures", exist_ok=True)
    train = np.squeeze(train)
    test = np.squeeze(test)
    plt.figure(dpi=300, figsize=(10, 6))
    plt.plot(range(len(train)), train, label="training set")
    plt.plot(range(len(test)), test, label="test set")
    plt.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel(f"{name}")
    plt.savefig(f"./figures/{name}.png")


def plot_cost(train):


    os.makedirs("./figures", exist_ok=True)
    train = np.squeeze(train)
    plt.figure(dpi=300, figsize=(10, 6))
    plt.plot(range(len(train)), train)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Best Cost")
    plt.savefig(f"./figures/cost_function.png")


def main():

    costs = []
    train_acc = []
    test_acc = []

    x_train, y_train, x_test, y_test = load_data()

    x_train = x_train.map(normalize)
    x_test = x_test.map(normalize)

    y_train = y_train.map(one_hot_matrix)
    y_test = y_test.map(one_hot_matrix)

    parameters = initialize_parameters()

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    number_of_epochs = 200

    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    minibatch_size = 32

    train_data = tf.data.Dataset.zip((x_train, y_train))
    test_data = tf.data.Dataset.zip((x_test, y_test))

    train_minibatches = train_data.batch(minibatch_size)
    test_minibatches = test_data.batch(minibatch_size)

    trainable_variables = [W1, b1, W2, b2, W3, b3]

    optimizers = tf.keras.optimizers.Adam(0.0001)
    m = train_data.cardinality().numpy()
    for epoch in range(number_of_epochs):

        train_accuracy.reset_states()
        test_accuracy.reset_states()
        epoch_cost = 0

        for (minibatch_x, minibatch_y) in train_minibatches:

            with tf.GradientTape() as tape:

                logits = forward_propagation(tf.transpose(minibatch_x), parameters)

                cost = compute_cost(tf.transpose(minibatch_y), logits)

            grads = tape.gradient(cost, trainable_variables)

            optimizers.apply_gradients(zip(grads, trainable_variables))

            train_accuracy.update_state(minibatch_y, tf.transpose(logits))

            epoch_cost += cost

        epoch_cost /= m

        costs.append(epoch_cost)
        train_acc.append(train_accuracy.result())
        for (minibatch_x, minibatch_y) in test_minibatches:

            z3 = forward_propagation(tf.transpose(minibatch_x), parameters)

            test_accuracy.update_state(minibatch_y, tf.transpose(z3))

        test_acc.append(test_accuracy.result())


        if epoch % 10 == 0:

            print("epoch: {}, cost: {}, train accuracy: {}, test accuracy: {}".format(epoch, epoch_cost, train_accuracy.result(), test_accuracy.result()))

    plot(train_acc, test_acc, "accuracy")
    plot_cost(costs)

    return parameters


if __name__ == "__main__":
    params = main()
