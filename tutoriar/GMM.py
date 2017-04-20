import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import *
from mpl_toolkits.mplot3d import Axes3D
from math import e as E

def gmm():
    # distribution 1
    x1size = 1000
    x2size = 1000
    x1 = np.random.normal(loc = 3.0,scale=1.5,size=x1size)
    x2 = np.random.normal(loc = 6.0,scale=2.5,size=x2size)
    y1 = np.zeros(x1size)
    y2 = np.ones(x2size)
    X = np.hstack((x1,x2))
    y = np.hstack((y1,y2))
    # initial models
    loc1 = 1.0
    scale1 = 2.0
    loc2 = 10.0
    scale2 = 2.0
    while True:
        # E step
        tx1 = []
        tx2 = []
        from scipy.stats import norm
        d1 = norm(loc1,scale1)
        d2 = norm(loc2,scale2)
        for x in list(X):
            if d1.pdf(x) > d2.pdf(x):
                tx1.append(x)
            else:
                tx2.append(x)
        # M step
        tx1 = np.array(tx1)
        tx2 = np.array(tx2)
        if np.abs(np.mean(tx2) - loc2) <= 0.0001 and np.abs(np.mean(tx1) - loc1) <= 0.0001:
            break
        loc1 = np.mean(tx1)
        loc2 = np.mean(tx2)
        scale1 = np.var(tx1)
        scale2 = np.var(tx2)
        print("------")
        print(loc1,scale1)
        print(loc2,scale2)
    print("------")
    print(loc1, scale1)
    print(loc2, scale2)
    g = GaussianMixture(n_components=2)
    g.fit(X.reshape((X.shape[0],1)),y)
    print(g.means_ )
    print(g.covariances_)


def test1():
    # distribution 1
    mean1 = (1,2)
    cov1 = [[1,0],[0,1]]
    data = np.random.multivariate_normal(mean1,cov1,(1,200))
    x1 = []
    y1 = []
    x = []
    y = []
    for a in data[0]:
        x1.append(a[0])
        y1.append(a[1])
        x.append(a[0])
        y.append(a[1])
    # distribution 2
    mean1 = (1.3,2.2)
    cov1 = [[1,0],[0,1]]
    data = np.random.multivariate_normal(mean1,cov1,(1,200))
    x2 = []
    y2 = []
    for a in data[0]:
        x2.append(a[0])
        y2.append(a[1])
        x.append(a[0])
        y.append(a[1])

    gmm = GaussianMixture()
    gmm.fit(x,y)
    # plt.scatter(x1,y1,s=20,c='red')
    # plt.scatter(x2,y2,s=20,c='blue')
    # plt.show()

def test2():
    fig  = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-3,3,0.25)
    Y = np.arange(-3,3,0.25)
    X,Y = np.meshgrid(X,Y)
    Z = E**(-X**2-Y**2)- E**(-(X-1)**2-(Y-1)**2)-E**(-(X+1)**2-(Y+1)**2)+E**(-(X-2)**2-Y**2)
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    ax.scatter(0,0,c="red")
    ax.scatter(2,0, c="red")
    ax.scatter(1,1, c="yellow")
    ax.scatter(1,-1,c="yellow")
    plt.show()




if __name__ == '__main__':
    gmm()
    # generatedata()
