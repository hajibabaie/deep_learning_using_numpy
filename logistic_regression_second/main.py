from Deep_Learning.logistic_regression_second.solution_method import LogisticRegression


def main():

    logistic_regression = LogisticRegression("train_catvnoncat",
                                             "test_catvnoncat",
                                             learning_rate=0.001,
                                             max_iteration=10000)

    results = logistic_regression.run()

    return results



if __name__ == "__main__":

    out = main()