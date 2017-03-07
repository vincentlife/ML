from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualizeDecesionTree():
    iris = load_iris()
    X = iris.data
    y = iris.target
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file=dot_data,feature_names=iris.feature_names,class_names=iris.target_names,
                         filled=True,rounded=True,impurity=False)
    graph1 = pydot.graph_from_dot_data(dot_data.getvalue())[0]
    graph1.write_pdf("iris.pdf")

def testPipeLine():
    from sklearn import pipeline

def irisVisualization():
    sns.set(style="white", color_codes=True)
    irisdata = load_iris()
    iris = pd.DataFrame(irisdata.data, columns=irisdata.feature_names)
    iris['Species'] = pd.Categorical.from_codes(irisdata.target, irisdata.target_names)
    # sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    # pandas plot
    # iris.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)")
    # iris.boxplot(by="Species", figsize=(12, 6))
    from pandas.tools.plotting import andrews_curves,parallel_coordinates
    andrews_curves(iris, "Species")
    parallel_coordinates(iris, "Species")


    # seaborn plot
    sns.jointplot(x="sepal length (cm)", y="sepal width (cm)", data=iris, size=5)
    sns.FacetGrid(iris, hue="Species", size=5) \
        .map(plt.scatter, "sepal length (cm)", "sepal width (cm)").add_legend() # sns.kdeplot
    sns.boxplot(x="Species", y="sepal length (cm)", data=iris)
    sns.violinplot(x="Species", y="sepal length (cm)", data=iris, size=6)
    sns.pairplot(iris, hue="Species", size=3)
    # sns.plt.show()



def testPreProc():

    iris = load_iris()
    # 无量纲化使不同规格的数据转换到同一规格。常见的无量纲化方法有标准化和区间缩放法。
    # 标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下。
    # 归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，
    # 拥有统一的标准，也就是说都转化为“单位向量”。
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    MinMaxScaler().fit_transform(iris.data)
    StandardScaler().fit_transform(iris.data)

    # 二值化，阈值设置为3，返回值为二值化后的数据
    from sklearn.preprocessing import Binarizer
    Binarizer(threshold=3).fit_transform(iris.data)

    # 哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
    from sklearn.preprocessing import OneHotEncoder
    OneHotEncoder().fit_transform(iris.target.reshape((-1, 1)))

    # 缺失值计算，返回值为计算缺失值后的数据
    # 参数missing_value为缺失值的表示形式，默认为NaN
    # 参数strategy为缺失值填充方式，默认为mean（均值）
    from numpy import vstack, array, nan
    from sklearn.preprocessing import Imputer
    Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))

    # 数据变换
    # 多项式变换
    from sklearn.preprocessing import PolynomialFeatures  # 多项式转换 #参数degree为度，默认值为
    PolynomialFeatures().fit_transform(iris.data)
    # 自定义转换函数为对数函数的数据变换 #第一个参数是单变元函数
    from numpy import log1p
    from sklearn.preprocessing import FunctionTransformer
    FunctionTransformer(log1p).fit_transform(iris.data)

    # 特征选择之filter
    # 方差选择法，返回值为特征选择后的数据 #参数threshold为方差的阈值
    from sklearn.feature_selection import VarianceThreshold
    VarianceThreshold(threshold=3).fit_transform(iris.data)
    # 选择K个最好的特征，返回选择特征后的数据
    # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
    # 输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
    # 第二个参数k为选择的特征个数
    from sklearn.feature_selection import SelectKBest
    from scipy.stats import pearsonr
    # 评价函数为 pearsonr 相关系数
    SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
    # 评价函数为 卡方检验函数
    from sklearn.feature_selection import chi2
    SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
    # 互信息法
    # from minepy import MINE
    # # 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
    # def mic(x, y):
    #     m = MINE()
    #     m.compute_score(x, y)
    #     return (m.mic(), 0.5)
    # SelectKBest(lambda X, Y: array(map(lambda x: mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

    # 特征选择之wrapper
    # 递归特征消除法，返回特征选择后的数据
    # 参数estimator为基模型
    # 参数n_features_to_select为选择的特征个数
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)

    # 特征选择之embedded
    # 使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。
    # 使用feature_selection库的SelectFromModel类结合带L1惩罚项的逻辑回归模型，来选择特征：
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    # 带L1惩罚项的逻辑回归作为基模型的特征选择
    SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)

    # L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。
    # 故可结合L2惩罚项来优化。若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，
    # 将这一集合中的特征平分L1中的权值

    # GBDT作为基模型的特征选择
    from sklearn.ensemble import GradientBoostingClassifier
    SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)


if __name__ == '__main__':
    irisVisualization()

