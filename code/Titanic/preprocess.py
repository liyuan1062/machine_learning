# coding=utf-8
# data analysis
import matplotlib
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
# virtualization
from matplotlib import pyplot as plt

# model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

train_df = pd.read_csv("../../data/Titanic/train.csv")
test_df = pd.read_csv("../../data/Titanic/test.csv")
combine = [train_df, test_df]

# print(train_df.head(10))
train_df.info()
print('*'*40)
print(train_df.describe(include=['O']))
print('*'*40)
test_df.info()
print('*'*40)
# x = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train_df['Age'].fillna(-20, inplace=True)
train_df['Age'].dropna(inplace=True)
# print(train_df.count(1))
# dy = train_df[train_df['Survived'] > 0]['Age']
# ddy = dy[dy.isnull() == False]
# dn = train_df[train_df['Survived'] == 0]['Age']
# ddn = dn[dn.isnull() == False]
# fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
# ax0.hist(ddy.values, 20)
# ax0.set_title("survived=1")
# ax1.hist(ddn.values, 20)
# ax1.set_title("survived=0")
# plt.ylim(0,150)
# plt.show()
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20).add_legend()
# sns.distplot(train_df['Age'])
# plt.show()
print("train null data summary:\n ", train_df.isnull().sum())
print("testing null data summary:\n ", test_df.isnull().sum())
datas = [train_df, test_df]
drop_columns = ['Name','Cabin', 'Ticket']
for data_df in datas:
    data_df.drop(drop_columns, axis=1, inplace=True)
    data_df['Embarked'].fillna(data_df['Embarked'].mode()[0], inplace=True)
    data_df['Age'].fillna(data_df['Age'].dropna().median(), inplace=True)
    data_df['Fare'].fillna(data_df['Fare'].dropna().median(), inplace=True)
    print("null data: -->\n", data_df.isnull().sum())

# create 2 new features
for data_df in datas:
    data_df['FamilySize'] = data_df['SibSp'] + data_df['Parch'] + 1
    data_df['IsAlone'] = 1
    data_df['IsAlone'].loc[data_df['FamilySize'] > 1] = 0
    print("mode IsAlone: ", data_df['IsAlone'].mode())

#
label = preprocessing.LabelEncoder()
oneHot = preprocessing.OneHotEncoder()
for data_df in datas:
    data_df['Sex_Code'] = label.fit_transform(data_df['Sex'])
    data_df['Embarked_Code'] = label.fit_transform(data_df['Embarked'])
    data_df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
    data_df.info()
    print(data_df.head())
# correlation heatmap
# sns.heatmap(train_df.corr(), square=True, vmax=1.0, annot=True)
# plt.show()

# model training
x_train = train_df.drop(['Survived', 'PassengerId'], axis=1)
y_train = train_df['Survived']
x_test = test_df.drop('PassengerId', axis=1)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_log = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train)*100, 2)
print(acc_log)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
acc_rf = round(rf.score(x_train, y_train)*100, 2)
print(acc_rf)
# print("test_log: ", y_pred, "test_rf:", y_pred_rf)

# svc = SVC()
# svc.fit(x_train, y_train)
# y_pred_svm = svc.predict(x_test)
# acc_svm = round(svc.score(x_train, y_train)*100, 2)
# print(acc_svm)
# submission = pd.DataFrame({"PassengerId": test_df['PassengerId'], "Survived": y_pred_rf})
# submission.to_csv("../../data/Titanic/test_predict_rf100.csv", index=False)

cv = ShuffleSplit(n_splits=10, train_size=0.8, test_size=None)
estimator = RandomForestClassifier(n_estimators=200)
train_size, train_scores, test_scores = learning_curve(estimator, x_train, y_train, train_sizes=np.linspace(.1,1,10))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_size, train_mean-train_std, train_mean+train_std, alpha=0.1, color='r')
plt.fill_between(train_size, test_mean-test_std, test_mean+test_std, alpha=0.1, color='g')
plt.plot(train_size, train_mean, 'o-', color='r', label='Train score')
plt.plot(train_size, test_mean, 'o-', color='g', label='CV score')
plt.legend(loc='best')
plt.show()

