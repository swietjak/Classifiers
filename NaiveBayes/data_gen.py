import numpy as np
import sklearn.datasets as dt
np.random.seed(420)


def parse_csv():
    train_features = np.loadtxt(f'./data/trainX.csv', delimiter=',')
    train_outputs = np.loadtxt(f'./data/trainD.csv', delimiter=',')
    train_data = np.column_stack(
        (np.transpose(train_features), np.transpose(train_outputs)))

    test_features = np.loadtxt(f'./data/testX.csv', delimiter=',')
    test_outputs = np.loadtxt(f'./data/testD.csv', delimiter=',')
    test_data = np.column_stack(
        (np.transpose(test_features), np.transpose(test_outputs)))
    return train_data, test_data


def generate_normal_dataset():
    mi1 = [0, 0]
    sigma1 = [[2, -1], [-1, 2]]

    mi2 = [2, 2]
    sigma2 = [[1, 0], [0, 1]]

    A1_data = np.random.multivariate_normal(mi1, sigma1, 100)

    A2_data = np.random.multivariate_normal(mi2, sigma2, 100)

    X = np.vstack((A1_data, A2_data))
    y = np.concatenate((np.ones(100), np.zeros(100)))
    data = np.column_stack((X, y))
    np.random.shuffle(data)

    return data[:100], data[100:]


def generate_not_normal_dataset():
    n_features = 11
    n_samples = 100
    A1_data = np.random.exponential(size=(n_samples, n_features))

    A2_data = np.random.exponential(size=(n_samples, n_features))

    X = np.vstack((A1_data, A2_data))
    y = np.concatenate((np.ones(100), np.zeros(100)))
    data = np.column_stack((X, y))
    np.random.shuffle(data)

    return data[:100], data[100:]
