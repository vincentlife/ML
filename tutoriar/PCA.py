import numpy as np

def KLTransform(l, dimen):
    classNum = len(l)
    featureNum = len(l[0][0])

    # translation
    meanVector = np.zeros(featureNum)
    c = 0
    for classes in l:
        tvector = np.zeros(featureNum)
        for t in classes:
            tvector = tvector + t
        meanVector = meanVector + tvector / len(classes)
    meanVector = meanVector / len(l)

    # get SwMatrix
    SwMatrix = np.zeros([featureNum, featureNum])
    for classs in l:
        a = np.mat(classs)
        s = np.zeros([featureNum, featureNum])
        for t in a:
            t = t - meanVector
            s = s + t.getT() * t
        s = s / a.shape[0]
        SwMatrix = SwMatrix + s
    SwMatrix = SwMatrix / featureNum

    # get dimension reduct matrix
    eigenvalue, eigenvec = np.linalg.eig(SwMatrix)
    # print(eigenvalue[0]*eigenvec[ :,0])
    # print(r*eigenvec[ :,0])
    tlist = []
    for i in range(eigenvalue.shape[0]):
        tlist.append((eigenvalue[i], eigenvec[:,i]))
    sorted(tlist, key=lambda x: x[0], reverse=True)
    rdmatrix = tlist[0][1]
    if dimen>=2:
        for i in range(1,dimen):
            rdmatrix = np.concatenate((rdmatrix,tlist[i][1]),axis=1)

    print(rdmatrix)

    # reduct dimension
    res = []
    for classes in l:
        a = np.mat(classes)*rdmatrix
        res.append(a.tolist())
    return res

# a = [[-5,-5],[-5,-4],[-4,-5],[-5,-6],[-6,-5]]
# b = [[5,5],[5,6],[6,5],[5,4],[4,5]]
a = [[0,0,0],[2,0,0],[2,0,1],[1,2,0]]
b = [[0,0,1],[0,1,0],[0,-2,1],[1,1,-2]]
res = KLTransform([a, b], 1)
for i in res:
    print(i)