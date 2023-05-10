import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import sklearn.datasets as dt
from utils import bayes_classifier_normal, bayes_classifier_parzen
from data_gen import parse_csv, generate_not_normal_dataset, generate_normal_dataset

train_data, test_data = parse_csv()

normal_bayes = bayes_classifier_normal(train_data, test_data)
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_data[:, -1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset')
plt.savefig("test.png")
plt.clf()
plt.scatter(test_data[:, 0], test_data[:, 1], c=normal_bayes)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Calculated by Normal Bayes')
plt.savefig("normal_bayes.png")
normal_bayes_conf = confusion_matrix(test_data[:, -1], normal_bayes)
acc = (normal_bayes_conf[0][0] + normal_bayes_conf[1]
       [1]) / np.sum(normal_bayes_conf)
print(f"normal_bayes accuracy: {acc}")
plt.clf()
plt.figure(figsize=(10, 7))
sn.heatmap(normal_bayes_conf, annot=True,
           cmap=sn.cubehelix_palette(as_cmap=True))
plt.savefig("normal_bayes_conf.png")
plt.clf()

benchmark_window_size = 1 / (len(train_data) ** (1/2))

window_sizes = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8,
                1, 2, 4, 6, 8, 9, 30, benchmark_window_size]
best_output_params = [0, [], []]
for size in window_sizes:
    parzen_bayes = bayes_classifier_parzen(
        train_data, test_data, h=size)
    parzen_bayes_conf = confusion_matrix(
        test_data[:, -1], parzen_bayes)
    acc = (parzen_bayes_conf[0][0] + parzen_bayes_conf[1]
           [1]) / np.sum(parzen_bayes_conf)
    if acc > best_output_params[0]:
        best_output_params[0] = acc
        best_output_params[1] = parzen_bayes
        best_output_params[2] = parzen_bayes_conf
    print(
        f"window size: {size}, acc: {acc}")

plt.scatter(test_data[:, 0], test_data[:, 1], c=best_output_params[1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Calculated by Parzen Windows')
plt.savefig("parzen_windows.png")
plt.clf()

plt.figure(figsize=(10, 7))
sn.heatmap(best_output_params[2], annot=True,
           cmap=sn.cubehelix_palette(as_cmap=True))
plt.savefig("parzen_window_conf.png")
plt.clf()
