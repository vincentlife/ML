import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import plot_decision_regions

def generateCircleData():
    r1 = 2
    r2 = 3
    X1 = np.random.random(200)*r1*2+3
    X2 = np.random.random(300)*r2*2+2
    # X1 = np.arange(3,3+2*r1,0.02)
    # X2 = np.arange(2,2+2*r2,0.02)
    bias1 = np.random.normal(0,0.1,200)
    bias2 = np.random.normal(0, 0.1, 300)
    Y1 = np.sqrt(r1*r1 - (X1-3-r1)**2)+5+bias1
    Y2 = np.sqrt(r2*r2 - (X2-2-r2)**2)+5+bias2
    # plt.scatter(X1,Y1,c='blue')
    # plt.scatter(X2,Y2,c='red')
    # plt.show()
    return X1,Y1,X2,Y2

def testSVMkernel():
    X1,Y1,X2,Y2 = generateCircleData()
    X = np.vstack((np.hstack((X1,X2)),np.hstack((Y1,Y2)))).T
    y = np.hstack((np.zeros(X1.shape[0]),np.ones(X2.shape[0])))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    from sklearn import svm
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train,y_train)
    plot_decision_regions(X_test,y_test,clf)

if __name__ == '__main__':
    testSVMkernel()

