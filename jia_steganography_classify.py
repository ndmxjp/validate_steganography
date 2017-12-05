import sklearn
from sklearn import svm
from sklearn import neural_network
import numpy as np
from sklearn.model_selection import KFold
from sklearn import datasets
import matplotlib.pyplot as plt

model   = sklearn.neural_network.MLPClassifier()

def run(model, X, y, title="", n_splits=10, random_state=None):
    print("====== %s ======" % title)
    kf = KFold(n_splits, random_state=random_state, shuffle=True)
    accuracy_sum = 0.0
    sum_count    = 0
    for train_index, test_index in kf.split(X):
        sum_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_ = model.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(y_test, y_)
        accuracy_sum += accuracy
        # print("true y: %s " % y_test)
        # print("predict y: %s " % y_)
        print("Accuracy: %f" % accuracy)

    average_accuracy = accuracy_sum / sum_count
    print("[%s] Average Accuracy: %f" % (title, average_accuracy))
    return average_accuracy


if False:
    iris = datasets.load_iris()
    run(
        svm.SVC(random_state=931),
        iris.data,
        iris.target,
        title="Non-Linear SVM",
        random_state = 101
    )

    run(
        sklearn.neural_network.MLPClassifier(random_state=3290),
        iris.data,
        iris.target,
        title="MLP",
        random_state=101
    )

if False:
    run(
        sklearn.neural_network.MLPClassifier(random_state=3290),
        features,
        labels,
        title="MLP",
        random_state=101
    )

if True:
    # print('./multiplication-reference-feature/f5_reference_all_mult_%.1f.csv' % 0.9)
    # exit()
    embed_rates       = []
    average_accuracys = []
    for embed_rate in np.arange(1, 10 +1) / 10:
        print("Loading(%.1f)..." % embed_rate)
        features = []
        labels   = []
        with open('./jia-feature/jia_0.0_feature.csv', 'r') as f:
            for line in f:
                feature = np.array([float(e) for e in line.split(',')])
                features.append(feature)
                labels.append(0)

        with open('./jia-feature/jia_%.1f_feature.csv' % embed_rate, 'r') as f:
            for line in f:
                feature = np.array([float(e) for e in line.split(',')])
                features.append(feature)
                labels.append(1)

        features = np.array([f / np.linalg.norm(f) for f in features])

        features = np.array(features)
        labels   = np.array(labels)
        print("Loaded!")

        average_accuracy = run(
            sklearn.neural_network.MLPClassifier(random_state=3290),
            # svm.SVC(random_state=931),
            features,
            labels,
            title="MLP",
            random_state=101
        )
        embed_rates.append(embed_rate)
        average_accuracys.append(average_accuracy)

    print(embed_rates)
    print(average_accuracys)

    plt.ylim(0.0, 1.0)
    plt.plot(embed_rates, average_accuracys)
    plt.show()
