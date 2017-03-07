import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import *
from mpl_toolkits.mplot3d import Axes3D
from math import e as E
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
    print(gmm.get_params())
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
    test2()
