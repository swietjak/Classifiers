import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate as integrate
from sklearn.metrics import confusion_matrix
import seaborn as sn
import sklearn.datasets as dt


def parzen_density_estimation(x, data, h):
    n = data.shape[0]  # number of samples in the dataset
    d = data.shape[1]  # number of features in the dataset
    hn = h / n**(1/2)
    # Compute the kernel function for each data point
    # kernel = np.exp(-np.sum((data - x)**2, axis=1) /
    #                 (2 * h**2)) / (np.sqrt(2 * np.pi) * h)**d

    kernel = np.exp(-np.sum((data - x)**2, axis=1) / hn) / \
        (np.sqrt(2 * np.pi) * hn) ** d

    density = np.sum(kernel) / n

    return density


def bayes_classifier_parzen(train_data, test_data, h):
    class_labels = np.unique(train_data[:, -1])

    filters_0 = train_data[:, -1] == class_labels[0]
    filters_1 = train_data[:, -1] == class_labels[1]

    X_0 = train_data[filters_0]
    X_1 = train_data[filters_1]

    results = []

    for i in range(test_data.shape[0]):
        x = test_data[i]
        likelihood_0 = parzen_density_estimation(x, X_0, h)
        likelihood_1 = parzen_density_estimation(x, X_1, h)

        if likelihood_0 > likelihood_1:
            results.append(class_labels[0])
        else:
            results.append(class_labels[1])

    return results


def bayes_classifier_normal(train_data, test_data):
    n = train_data.shape[1] - 1
    class_labels = np.unique(train_data[:, -1])

    filters_0 = train_data[:, -1] == class_labels[0]
    filters_1 = train_data[:, -1] == class_labels[1]

    X_0 = train_data[filters_0]
    X_1 = train_data[filters_1]

    mean_0 = np.mean(X_0[:, :n], axis=0)
    var_0 = np.var(X_0[:, :n], axis=0)

    mean_1 = np.mean(X_1[:, :n], axis=0)
    var_1 = np.var(X_1[:, :n], axis=0)
    print(X_1.shape, mean_0.shape, var_1.shape)
    results = []

    for x in test_data:
        pdf_0 = norm.pdf(x[:-1], mean_0, var_0)
        pdf_1 = norm.pdf(x[:-1], mean_1, var_1)
        likelihood_0 = np.prod(pdf_0)
        likelihood_1 = np.prod(pdf_1)

        if likelihood_0 > likelihood_1:
            results.append(class_labels[0])
        else:
            results.append(class_labels[1])
    return results
