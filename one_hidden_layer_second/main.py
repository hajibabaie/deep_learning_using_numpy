from Deep_Learning.one_hidden_layer_second.solution_method import OneHiddenLayer


def main():

    NN = OneHiddenLayer("train_catvnoncat",
                        "test_catvnoncat",
                        2, 0.0001, 10000)

    result = NN.run()

    return result


if __name__ == "__main__":

    parameters = main()
