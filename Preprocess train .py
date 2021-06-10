import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy.sparse import data
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import random

import matplotlib.pyplot as plt

plate = []
dfList = {"Infant": [0, 1], "Toddler": [2, 4], "Child": [5, 12, 'red'], "Teen": [
    13, 19, 'yellow'], "Adult": [20, 39, 'blue'], "Middle Age": [40, 55, 'green'], "Senior": [56, 100, 'purple']}

palette = {"Child": 'red', "Teen": 'yellow', "Adult": 'blue',
           "Middle Age":  'green', "Senior":  'purple'}


def data_filter(X, feat):
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='mean', verbose=1)
    X = imputer.fit_transform(X)

    X = pd.DataFrame(X, columns=feat)

    return X


def LogistModel(X, Y, feat):
    global curcol
    X = data_filter(X, feat)

    model = LogisticRegression(max_iter=10000000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=42, test_size=0.3)

    model.fit(X_train, y_train)

    return model.score(X_test, y_test)


def KNN(X, Y, subject):
    global curcol, feat
    X = data_filter(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=42, test_size=0.3)

    n, m = X_train.shape

    # plot n_neighbours - accuracy graph
    # n_neighbours : 1 - 21
    nn = []
    # score performance
    perf = []
    ma = -1

    for k in range(1, min(n, 25), 2):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test)

        score = metrics.accuracy_score(y_test, Y_pred)
        nn.append(k)
        perf.append(score)
        ma = max(ma, score)

    plt.plot(nn, perf, c=palette[subject], label=subject)
    plt.legend()
    print(ma)


dismiss = ['site', 'dx', 'dsm', 'handedness', 'volume_Right-non-WM-hypointensities', 'volume_Right-WM-hypointensities',
           'volume_Left-WM-hypointensities', 'volume_Left-non-WM-hypointensities', 'qc_rater_1', 'qc_rater_4', 'qc_anat_rater_3', 'qc_anat_rater_2']


# feature selection functions

df = pd.read_csv('ABIDE_Complete_2017.csv')
# print(df.shape)

n, m = df.shape

col = ['subjectAge', 'researchGroup']
ind = df.columns.get_loc('L_superior_frontal_gyrus')


for i in range(ind, m):

    if df.columns[i] in dismiss:
        continue
    col.append(df.columns[i])
    # print(max(df[col[-1]].values))

df = df[col]

print(df.shape)

X = df[col[2:]]


# print(type(df['subjectAge'][9]))


for i in dfList:

    dfList[i] = df.loc[(df['subjectAge'] > dfList[i][0]) &
                       (df['subjectAge'] <= dfList[i][1])]

f = open("performance.txt", 'w')

for i in dfList:
    print(i)
    curcol = col[:]
    df = dfList[i]

    # print("SHAPE : ", df.shape)

    n, m = df.shape
    if n == 0:
        continue
    Y = df['researchGroup'].values
    X = df[col[2:]]

    qconstant_filter = VarianceThreshold(threshold=0.01)
    qconstant_filter.fit(X)

    qconstant_columns = [column for column in X.columns
                         if column not in X.columns[qconstant_filter.get_support()]]

    print(len(qconstant_columns))

    for k in qconstant_columns:

        curcol.remove(k)
    X = df[curcol[2:]]
    X = data_filter(X, curcol[2:])
    orig = []
    for k in range(1, 11):
        model = LogisticRegression(max_iter=10000000)

        rfe = RFE(model, k)
        fit = rfe.fit(X, Y)
        print("Selected features : {} ".format(fit.support_))
        arr = fit.ranking_

        if k == 1:
            orig = arr
            continue
        if arr == orig:
            print("YAYYYY")

    print("Feature ranking : {}".format(arr))
    print("Size : ", len(arr))
    print("type: ", type(arr))

    # varying number of features being selected
    ma = - 1
    noF = 0
    Fcol = curcol[2:]
    m = len(Fcol)

    arr = list(arr)
    # print(arr.sort())
    zipped_lists = zip(arr, Fcol)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    arr, Fcol = [list(tuple) for tuple in tuples]
    num_f = []
    perf = []

    optK = 0
    optFeat = []
    for r in range(1, m, 25):
        feat = []
        for k in range(0, r + 1):
            feat.append(Fcol[k])
        X = df.loc[:, feat].values

        score = LogistModel(X, Y, feat)
        print(score)
        ma = max(score, ma)
        if score == ma:
            noF = r
            optFeat = feat
            optK = r + 1
        perf.append(score)
        num_f.append(r)
    plt.plot(num_f, perf, c=palette[i], label=i)
    plt.legend()

    f.write(i + " | max accuracy :   " + str(ma) + '\n' +
            str(optK) + " Features " + "\n" + str(optFeat))
    f.write("\n\n\n")


plt.title("Number of selected features vs. Performance")
plt.xlabel("No. of selected features ")
plt.ylabel("Accuracy score")
plt.savefig("Logreg-Feature-Acc.png")
plt.show()
