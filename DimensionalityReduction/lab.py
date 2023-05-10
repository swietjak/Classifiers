import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def generateClasses():
    class_0 = np.zeros(shape=(200, 50))
    class_1 = np.zeros(shape=(200, 50))
    for i in range(200):
        class_0[i][0] = np.random.normal(loc=5, scale=3)
        class_0[i][1] = np.random.normal(loc=10, scale=0.5)
        class_0[i][2] = np.random.uniform(low=0, high=10)
        class_0[i][3] = np.random.exponential(scale=2)
        class_0[i][4] = np.random.gamma(shape=4, scale=3)
        class_1[i][0] = np.random.normal(loc=10, scale=2)
        class_1[i][1] = np.random.normal(loc=3, scale=1)
        class_1[i][2] = np.random.uniform(low=6, high=24)
        class_1[i][3] = np.random.exponential(scale=7)
        class_1[i][4] = np.random.gamma(shape=2, scale=8)

    for i in range(200):
        class_0[i][5:] = [np.random.uniform(low=0, high=15) for j in range(45)]
        class_1[i][5:] = [np.random.uniform(low=0, high=15) for j in range(45)]

    classes = []
    for i in range(200):
        classes.append((class_0[i], 0))
        classes.append((class_1[i], 1))

    return classes


def parseIndicies(feature_names):
    return [int(i.replace("x", "")) for i in feature_names]


def load_csv():
    features = np.loadtxt("cancer_X.csv", delimiter=",")
    labels = np.loadtxt("cancer_D.csv", delimiter=",")
    print(np.transpose(labels).shape, features.shape)
    transposed_features = np.transpose(features)
    return [(transposed_features[i], labels[i]) for i in range(len(transposed_features))]


classes = generateClasses()
file_data = load_csv()
train, test = train_test_split(file_data, test_size=0.5)

train_labels = np.array([t[1] for t in train])
train_features = np.array([t[0] for t in train])
test_labels = np.array([t[1] for t in test])
test_features = np.array([t[0] for t in test])

# feature selection
selected_features = SelectKBest(k=2).fit_transform(
    train_features, train_labels)
plt.scatter(selected_features[:, 0],
            selected_features[:, 1], cmap='viridis', c=train_labels)
plt.savefig('selected.png')
plt.clf()

# feature extraction
transformer = PCA(n_components=2)
train_features_transformed = transformer.fit_transform(train_features)
# print(train_features_transformed.shape)
plt.scatter(train_features_transformed[:, 0],
            train_features_transformed[:, 1], cmap='viridis', c=train_labels)
plt.savefig('extracted.png')
plt.clf()

svm_classifier_selection = svm.SVC()
selected_features3 = SelectKBest(k=10).fit(train_features, train_labels)
selected_indicies = parseIndicies(selected_features3.get_feature_names_out())
svm_classifier_selection.fit(
    train_features[:, selected_indicies], train_labels)
selection_classifier_results = svm_classifier_selection.predict(
    test_features[:, selected_indicies])
print(accuracy_score(test_labels, selection_classifier_results))

svm_classifier_extraction = svm.SVC()
pca = PCA(n_components=3)
extracted_train_features3 = pca.fit_transform(train_features, train_labels)
extracted_test_features3 = pca.fit_transform(test_features, test_labels)
svm_classifier_extraction.fit(extracted_train_features3, train_labels)
extraction_classifier_results = svm_classifier_extraction.predict(
    extracted_test_features3)
print(accuracy_score(test_labels, extraction_classifier_results))

random_indicies = np.random.randint(0, 50, 3)
random_train_features = train_features[:, random_indicies]
random_test_features = test_features[:, random_indicies]

svm_classifier_random = svm.SVC()
svm_classifier_random.fit(random_train_features, train_labels)
random_classifier_results = svm_classifier_random.predict(random_test_features)
print(accuracy_score(test_labels, random_classifier_results))

svm_classifier_combined = svm.SVC()
svm_classifier_combined.fit(train_features, train_labels)
combined_results = svm_classifier_combined.predict(test_features)
print(accuracy_score(test_labels, combined_results))


def write_to_file(line):
    with open('accuracies.txt', "a+") as f:
        f.write(line + "\n")


def selection(k):
    svm_classifier_selection = svm.SVC()
    selected_features3 = SelectKBest(k=k).fit(train_features, train_labels)
    selected_indicies = parseIndicies(
        selected_features3.get_feature_names_out())
    svm_classifier_selection.fit(
        train_features[:, selected_indicies], train_labels)
    selection_classifier_results = svm_classifier_selection.predict(
        test_features[:, selected_indicies])
    write_to_file(
        f"Selection k: {k} acc: {accuracy_score(test_labels, selection_classifier_results)}")


def extraction(n):
    svm_classifier_extraction = svm.SVC()
    pca = PCA(n_components=n)
    extracted_train_features3 = pca.fit_transform(train_features, train_labels)
    extracted_test_features3 = pca.fit_transform(test_features, test_labels)
    svm_classifier_extraction.fit(extracted_train_features3, train_labels)
    extraction_classifier_results = svm_classifier_extraction.predict(
        extracted_test_features3)
    write_to_file(
        f"Extraction n: {n} acc: {accuracy_score(test_labels, extraction_classifier_results)}")


for i in [2, 3, 5, 7, 10, 20]:
    selection(i)
    extraction(i)
