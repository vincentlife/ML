import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score

trainfile = r"D:\DateSet\titanic\train.csv"
testfile = r"D:\DateSet\titanic\test.csv"

def alphaversion():
    # Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
    train = pd.read_csv(trainfile).drop(["PassengerId","Name","Cabin","Ticket"],axis=1).dropna()
    train = train.reset_index(drop=True)


    # two ways to convert 1d array to 2d array
    # d = LabelEncoder().fit_transform(train["Sex"]).reshape(train["Sex"].shape[0],1)
    sexcode = LabelEncoder().fit_transform(train["Sex"])[np.newaxis].T
    embarkedcode = LabelEncoder().fit_transform(train["Embarked"])[np.newaxis].T
    pclasscode = np.array(train["Pclass"])[np.newaxis].T

    d = OneHotEncoder().fit_transform(np.hstack((sexcode, embarkedcode,pclasscode))).toarray()
    train_proc = train.drop(["Pclass","Sex","Embarked"],axis=1).join(pd.DataFrame(d))

    train_proc["Age"] = StandardScaler().fit_transform(train_proc["Age"])
    train_proc["Fare"] = StandardScaler().fit_transform(train_proc["Fare"])

    # transform TEST data
    test = pd.read_csv(testfile).drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
    test = test.fillna(test.mean()["Age"])
    sexcode = LabelEncoder().fit_transform(test["Sex"])[np.newaxis].T
    embarkedcode = LabelEncoder().fit_transform(test["Embarked"])[np.newaxis].T
    pclasscode = np.array(test["Pclass"])[np.newaxis].T

    d = OneHotEncoder().fit_transform(np.hstack((sexcode, embarkedcode, pclasscode))).toarray()
    test_proc = test.drop(["Pclass", "Sex", "Embarked"], axis=1).join(pd.DataFrame(d))

    test_proc["Age"] = StandardScaler().fit_transform(test_proc["Age"])
    test_proc["Fare"] = StandardScaler().fit_transform(test_proc["Fare"])

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()

    clf.fit(train_proc.drop(["Survived"],axis=1),train_proc["Survived"])

    r = pd.read_csv(r"D:\DateSet\titanic\gender_submission.csv")["Survived"]
    # print(accuracy_score(r, clf.predict(test_proc.values)))
    d1 = pd.DataFrame(pd.read_csv(testfile)["PassengerId"],columns=["PassengerId"])
    # dd.append(pd.DataFrame(clf.predict(test_proc.values), columns=["Survived"]), ignore_index=True)
    d2 = pd.DataFrame(clf.predict(test_proc.values), columns=["Survived"])
    dd = pd.concat([d1,d2],axis=1)
    dd.to_csv(r"D:\DateSet\titanic\titanic.csv",index=False)

def datastat():
    train_df = pd.read_csv(trainfile)
    test_df = pd.read_csv(testfile).drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    # the number of NaN axis=0代表函数应用于每一列
    print(train_df.apply(lambda x: sum(x.isnull()), axis=0))

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # print( list(filter(lambda a: a < 10, dataset.groupby(["Title"]).size())) )
        # print([x for x in dataset.groupby(["Title"]).size() if x<10])
        # l = dataset.groupby(["Title"]).size()
        # print(l[l<10])
        # for x,y in l.iteritems():
        #     if y<10:
        #         print(x)
    def Survived_factor(factor):
        return train_df[[factor, "Survived"]].groupby([factor], as_index=False).mean().sort_values(by='Survived', ascending=False)
    for x in ["Sex","Pclass","SibSp","Parch"]:
        print(Survived_factor(x))

def datavisualize():
    train_df = pd.read_csv(trainfile)
    g = sns.FacetGrid(train_df, col='Survived', row='Pclass')
    g.map(plt.hist, 'Age',alpha=.5, bins=20)
    sns.plt.show()

def betaversion():
    train_df = pd.read_csv(trainfile).drop(['Ticket', 'Cabin'], axis=1)
    test_df = pd.read_csv(testfile).drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False))



if __name__ == '__main__':
    datastat()
    # betaversion()