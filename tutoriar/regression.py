
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from functools import reduce

def generate2dData():
    x = np.arange(0,1,0.02)
    yt = norm.rvs(0,size=50,scale=0.5)
    y = np.array(list(map(lambda a,b: a+b**2,x,yt)))
    return x,y

def polynomialRegression():
    x,y = generate2dData()
    plt.scatter(x,y,s=5)
    degrees = [1,2,6]
    for d in degrees:
        clf = Pipeline([("poly",PolynomialFeatures(degree=d)),("linear",LinearRegression(fit_intercept=True))])
        clf.fit(x[:, np.newaxis], y)
        y_test = clf.predict(x[:, np.newaxis])
        plt.plot(x, y_test, linewidth=2)
    plt.grid()
    plt.legend(degrees, loc='upper left')
    plt.show()

def generate3dData():
    x1 = np.random.random(size=100)*100
    x2 = np.random.random(size=100)*100
    yt = norm.rvs(0,size=100,scale=10)
    y = np.array(list(map(lambda a : 1 + a, yt)))
    return np.mat(x1).T,np.mat(x2).T,np.mat(y).T


def closedformLR(X,y):
    # closed-form solution
    w = ((X.T*X).I)*X.T*y
    v = w.T.tolist()[0]
    return v

def GDLR(X,y):
    # gradient descent
    epsilon = 0.00000000001
    steplength = 0.0001
    h = np.mat([0]*X.shape[1])
    sample_num = X.shape[0]

    for i in range(100000):
        for x in X:
            next_h = h + steplength*((y[i % sample_num]-h*x.T)*x)
            if sum(map(lambda a,b : (a-b)*(a-b), next_h.tolist()[0] , h.tolist()[0] )) < epsilon:
                print("Already convergence with "+str(i)+" iterations")
                return next_h
            h = next_h
    return h

def SKLR(X,y):
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X,y)
    return lr.coef_


def testLinearRegression():
    x1,x2,y = generate3dData()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(np.array(x1.ravel()), np.array(x2.ravel()), np.array(y.ravel()), c="red")
    x1plot = np.arange(0,100,2)
    x2plot = np.arange(0,100,2)

    x0 = np.mat([1] * len(x1)).T
    X = np.concatenate((x1,x2,x0), axis=1)
    v1 = closedformLR(X,y)
    print("closed-form solution:")
    print(v1)
    y1plot = x1plot*v1[0]+x2plot*v1[1]+v1[2]

    # gradient descent
    v2 = GDLR(X, y).tolist()[0]
    y2plot = x1plot*v2[0]+x2plot*v2[1]+v2[2]
    print("gradient descent solution:")
    print(v2)

    # skleeran
    v3 = SKLR(X,y)[0]
    y3plot =x1plot*v3[0]+x2plot*v3[1]+v3[2]
    print("sklearn solution:")
    print(v3)


    x1plot, x2plot  = np.meshgrid(x1plot, x2plot)
    ax.plot_wireframe(x1plot, x2plot, y1plot,rstride=1,cstride=1,color="green")
    ax.plot_wireframe(x1plot, x2plot, y2plot, rstride=1, cstride=1, color="blue")
    ax.plot_wireframe(x1plot, x2plot, y3plot, rstride=1, cstride=1, color="yellow")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    testLinearRegression()