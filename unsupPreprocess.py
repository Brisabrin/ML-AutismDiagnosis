from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
le = preprocessing.LabelEncoder()


df = pd.read_csv('ABIDE_Complete_2017.csv')
# print(df.shape)


ind = df.columns.get_loc('L_superior_frontal_gyrus')
# print(df.shape)
ind2 = df.columns.get_loc('researchGroup')

Xarr = []

Xarr = list(df.columns)
Xarr = Xarr[ind: ind + 50]

df = df.apply(le.fit_transform)

X = df.iloc[:, ind:ind+50]

Y = df.iloc[:, ind2]

f = pd.DataFrame(X)
print(f)
f.insert(50, 'researchGroup', Y.values, True)

plt.figure(figsize=(12, 10))
cor = f.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()
cor_target = abs(cor['researchGroup'])

relevant_features = pd.Series(cor_target[cor_target < 1])

print(relevant_features)

# feature selection

# model = KNeighborsClassifier()

# rfe = RFE(model, 5)

# fit = rfe.fit(X, Y)

# arr = fit.ranking_

# NewList = []
# for i in range(len(arr)):

#     if arr[i] <= 3:
#         NewList.append(Xarr[i])


# X = df[NewList]


model = KNeighborsClassifier(n_neighbors=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=0, test_size=0.3)

print(type(X_train))

# model.fit(X_train, y_train)

# predicted = model.predict(X_test)
# print(predicted)

# print("Accuracy : {} ".format(metrics.accuracy_score(y_test, predicted)))
