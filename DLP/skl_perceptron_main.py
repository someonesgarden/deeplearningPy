from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from lib.util import plot_decisionregions

def main():
    # DATA #30% test / 70% training data =====
    iris = ds.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # STANDARD ===============================
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # FIT ====================================
    ppn = Perceptron(n_iter=40, eta0=0.01, random_state=0)
    ppn.fit(X_train_std, y_train)

    # PREDICTION =============================
    y_pred = ppn.predict(X_test_std)

    # ANALYSE ================================
    missed_samples = (y_test !=y_pred).sum()
    print "Misclassified : {0} ({1:.2f}%)".format(missed_samples,  (missed_samples*100/len(y_test)))
    print "Accuracy: {0:.2f}".format(accuracy_score(y_test,y_pred))

    # =============================
    resolution = 0.02
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    x1_min, x1_max = X_combined_std[:, 0].min() - 1, X_combined_std[:, 0].max() + 1
    x2_min, x2_max = X_combined_std[:, 1].min() - 1, X_combined_std[:, 1].max() + 1
    X1, X2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = ppn.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)

    plot_decisionregions(X_combined_std, X1, X2, Z, y_combined, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal weight [standardized]')
    plt.legend(loc='upper left')
    plt.show()