import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn import svm


def draw_confusion_matrix(expected_outputs, received_outputs, filename):
    conf = confusion_matrix(
        expected_outputs, received_outputs,)
    print(conf)
    acc = (conf[0][0] + conf[1]
           [1]) / np.sum(conf)
    print(
        f"{filename} accuracy: {acc} specificity: {conf[0][0]/(conf[0][0]+conf[0][1])} sensitivity: {conf[1][1]/(conf[1][0]+conf[1][1])}")
    plt.clf()
    plt.figure(figsize=(10, 7))
    sn.heatmap(conf, annot=True,
               cmap=sn.cubehelix_palette(as_cmap=True), fmt="g")
    plt.savefig(f"{filename}.png")
    plt.clf()

    return acc


dfa = pd.read_csv("./dataset3a.csv")
dfb = pd.read_csv("./dataset3b.csv")

print(f"A dataset shape: {dfa.shape}; B dataset shape: {dfb.shape}")

dfa["y"] = dfa["y"].astype("category")
dfb["y"] = dfb["y"].astype("category")

sn.pairplot(dfa[["x1", "x2", 'y']], hue="y")
plt.savefig("dfa_pairplot.png")
plt.clf()
sn.pairplot(dfb[["x1", "x2", "y"]], hue="y")
plt.savefig("dfb_pairplot.png")
plt.clf()

dfa.plot.scatter("x1", "x2", c="y", colormap="viridis")
plt.savefig("dfa_scatter.png")
plt.clf()
dfb.plot.scatter("x1", "x2", c="y", colormap="viridis")
plt.savefig("dfb_scatter.png")
plt.clf()

dfa.boxplot(column=["x1", "x2"])
plt.savefig("dfa_boxplot.png")
plt.clf()
dfb.boxplot(column=["x1", "x2"])
plt.savefig("dfb_boxplot.png")
plt.clf()

train_dfa, test_dfa = train_test_split(dfa, test_size=0.3)
train_dfb, test_dfb = train_test_split(dfb, test_size=0.3)

# LDA
lda_classifier_a = LDA()
lda_classifier_b = LDA()

lda_classifier_a.fit(train_dfa[['x1', 'x2']].to_numpy(), train_dfa["y"])
predicted_lda_a = lda_classifier_a.predict(test_dfa[['x1', 'x2']])
lda_classifier_b.fit(train_dfb[['x1', 'x2']].to_numpy(), train_dfb["y"])
predicted_lda_b = lda_classifier_b.predict(test_dfb[['x1', 'x2']])
print("lda", lda_classifier_a.predict_proba(train_dfa[['x1', 'x2']].to_numpy()),
      lda_classifier_b.predict_proba(train_dfa[['x1', 'x2']].to_numpy()))
draw_confusion_matrix(test_dfa['y'], predicted_lda_a, "lda_a")
draw_confusion_matrix(test_dfb['y'], predicted_lda_b, "lda_b")

# QDA
qda_classifier_a = QDA()
qda_classifier_b = QDA()

qda_classifier_a.fit(train_dfa[['x1', 'x2']].to_numpy(), train_dfa["y"])
predicted_qda_a = qda_classifier_a.predict(test_dfa[['x1', 'x2']])
qda_classifier_b.fit(train_dfb[['x1', 'x2']].to_numpy(), train_dfb["y"])
predicted_qda_b = qda_classifier_b.predict(test_dfb[['x1', 'x2']])
print("qda", qda_classifier_a.predict_proba(train_dfa[['x1', 'x2']].to_numpy()),
      qda_classifier_b.predict_proba(train_dfa[['x1', 'x2']].to_numpy()))
draw_confusion_matrix(test_dfa['y'], predicted_qda_a, "qda_a")
draw_confusion_matrix(test_dfb['y'], predicted_qda_b, "qda_b")
# SVM
svm_classifier_a = svm.SVC(probability=True)
svm_classifier_b = svm.SVC(probability=True)

svm_classifier_a.fit(train_dfa[['x1', 'x2']].to_numpy(), train_dfa["y"])
predicted_svm_a = svm_classifier_a.predict(test_dfa[['x1', 'x2']])
svm_classifier_b.fit(train_dfb[['x1', 'x2']].to_numpy(), train_dfb["y"])
predicted_svm_b = svm_classifier_b.predict(test_dfb[['x1', 'x2']])

print("svm", svm_classifier_a.predict_proba(train_dfa[['x1', 'x2']].to_numpy()),
      svm_classifier_b.predict_proba(train_dfa[['x1', 'x2']].to_numpy()))

draw_confusion_matrix(test_dfa['y'], predicted_svm_a, "svm_a")
draw_confusion_matrix(test_dfb['y'], predicted_svm_b, "svm_b")
