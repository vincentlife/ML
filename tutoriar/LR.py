import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from tools import plot_decision_regions

def logr2():
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    X = iris_data.data[:, [2, 3]]
    y = iris_data.target

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    s = lr.predict_proba(X_test_std[0, :])  # 查看第一个测试样本属于各个类别的概率
    print(s)
    plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    # df = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
    # df['species'] = pd.Categorical.from_codes(iris_data.target,iris_data.target_names)
    # X = df.ix[:,[2,3]]

def logisticRegression1():
    filename = r"D:\DateSet\binary.csv"
    df = pd.read_csv(filename)
    df.columns = ["admit", "gre", "gpa", "prestige"]
    # print(df.head())
    # print(df.describe())
    # print(pd.crosstab(df['admit'], df['prestige'], rownames=['admit']))
    # df.hist()
    # pl.show()
    dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')

    cols_to_keep = ["admit", "gre", "gpa"]

    data = df[cols_to_keep].join(dummy_ranks.ix[:,'prestige_2':])
    train_cols = data.columns[1:]

    # statsmodels

    # # sklearn logistic regression
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(data[train_cols],data['admit'],test_size=0.2,random_state=1)
    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression(C=1.0, random_state=0)
    # lr.fit(X_train,y_train)
    # print(lr.score(X_test,y_test))

if __name__ == '__main__':
    # logisticRegression1()
    logr2()
