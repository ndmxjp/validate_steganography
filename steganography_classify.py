import sklearn
from sklearn import svm
from sklearn import neural_network
import numpy as np
from sklearn.model_selection import KFold
from sklearn import datasets
import matplotlib.pyplot as plt




# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 1, 1, 0])

iris = datasets.load_iris()
X = iris.data
y = iris.target

# model = svm.SVC()
model   = sklearn.neural_network.MLPClassifier()

# def calc_accuracy(y, y_):


def run(model, X, y, title="", n_splits=10, random_state=None):
    print("====== %s ======" % title)
    kf = KFold(n_splits, random_state=random_state, shuffle=True)
    train_accuracy_sum = 0.0
    test_accuracy_sum = 0.0
    sum_count    = 0
    for train_index, test_index in kf.split(X):
        sum_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        train_pred      = model.predict(X_train)
        test_pred       = model.predict(X_test)
        train_accuracy  = sklearn.metrics.accuracy_score(y_train, train_pred)
        test_accuracy   = sklearn.metrics.accuracy_score(y_test, test_pred)
        train_accuracy_sum += train_accuracy
        test_accuracy_sum  += test_accuracy
        # print("true y: %s " % y_test)
        # print("predict y: %s " % y_)
        try:
            print("n_support_:", model.n_support_)
        except:
            pass
        print("Train Accuracy: %f" % train_accuracy, "Test Accuracy: %f" % test_accuracy)

    train_average_accuracy = train_accuracy_sum / sum_count
    test_average_accuracy  = test_accuracy_sum / sum_count
    print("[%s] Average Train Accuracy: %f" % (title, train_average_accuracy))
    print("[%s] Average Test Accuracy: %f" % (title, test_average_accuracy))
    return test_average_accuracy


if False:
    iris = datasets.load_iris()
    run(
        svm.SVC(random_state=931),
        iris.data,
        iris.target,
        title="Non-Linear MLP",
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

    algorithms = ["f5", "jstego", "outguess"]
    for algorithm in algorithms:
        embed_rates       = []
        average_accuracys = []
        take_num          = -1

        for embed_rate in np.arange(1, 10 +1) / 10:
            print("Loading(%.1f)..." % embed_rate)
            features = []
            labels   = []
            with open('./%s-multiplication-reference-feature/%s_reference_all_mult_0.0.csv' % (algorithm, algorithm), 'r') as f:
                for line in f:
                    feature = np.array([float(e) for e in line.split(',')])
                    features.append(feature[:take_num])
                    labels.append(0)

            with open('./%s-multiplication-reference-feature/%s_reference_all_mult_%.1f.csv' % (algorithm, algorithm, embed_rate), 'r') as f:
                for line in f:
                    feature = np.array([float(e) for e in line.split(',')])
                    features.append(feature[:take_num])
                    labels.append(1)

            features = np.array([f / np.linalg.norm(f) for f in features])
            # features = features / np.max(np.abs(features))
            labels   = np.array(labels)
            print("Loaded!")

            print('len(features):', len(features))
            average_accuracy = run(
                # svm.SVC(random_state=3290),
                sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (100,)),
                features,
                labels,
                title="MLP - %s" % algorithm,
                random_state=101
            )
            embed_rates.append(embed_rate)
            average_accuracys.append(average_accuracy)

        print(embed_rates)
        print(average_accuracys)

        plt.plot(embed_rates, average_accuracys, label = algorithm)

    plt.legend()
    plt.title("MLP h5")
    plt.xlabel("embed rate(%)")
    plt.ylabel("Accuracy(%)")
    plt.show()
