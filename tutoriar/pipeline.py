import numpy as np
import pandas as pd

def testPipeLine():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    from sklearn import tree
    clf1 = tree.DecisionTreeClassifier()
    clf1.fit(X_train,y_train)
    from sklearn.neighbors import KNeighborsClassifier
    clf2 = KNeighborsClassifier()
    clf2.fit(X_train,y_train)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test,clf1.predict(X_test)))
    print(accuracy_score(y_test,clf2.predict(X_test)))


if __name__ == '__main__':
    testPipeLine()
