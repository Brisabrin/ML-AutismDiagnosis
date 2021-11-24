import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy.sparse import data
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB


"""" 
logsitic regression 
knn 
random forest 
naive bayes 
decision trees 
"""
#function to write values to file 

plate = []
dfList = { "Child": [5, 12, 'red'], "Teen": [
    13, 19, 'yellow'], "Adult": [20, 39, 'blue']}

palette = {"Child": 'red', "Teen": 'yellow', "Adult": 'blue',
           "Middle Age":  'green', "Senior":  'purple'}

f = ''

def data_filter(X, feat):
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='mean', verbose=1)
    X = imputer.fit_transform(X)

    X = pd.DataFrame(X, columns=feat)

    return X



# new_knn_model = KNeighborsClassifier(**knn_gridsearch_model.best_params_)



def naive_bayes( X_train, X_test, y_train )  : 

    model = GaussianNB()
    model.fit(X_train , y_train)

    y_pred = model.predict(X_test)
    return y_pred

def Logistic_reg( X_train, X_test, y_train ) : 


    model = LogisticRegression(max_iter = 10000000) 
    model.fit(X_train , y_train)

    y_pred = model.predict(X_test)
    return y_pred 
    
        

def KNN_findparam(X, Y):
    global curcol, feat , f
    X = data_filter(X , feat)
    print("Sup dudes")

    # X_train, X_test, y_train, y_test = train_test_split(
        # X, Y, random_state=42, test_size=0.3) 

    n, m = X.shape 


    knn =  KNeighborsClassifier()
    #iterates through to find the optimal number of nearest neigbours
    param = { 'n_neighbors' : [ k  for k in range(1, min(n, 25), 2)]}

    model = GridSearchCV(knn , param , cv = 5 )
    model.fit(X, Y) 


    f.write("best params n_neighbors {}".format(model.best_params_))

    return model.best_params_['n_neighbors']




def KNN_train_model( optparam , X_train, X_test, y_train )  :

    model =  KNeighborsClassifier(n_neighbors=optparam)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    return y_pred


def random_forest_findparam( X, Y) :  
    global feat  , curcol 

    X = data_filter(X , feat)

    param_grid  =  {"n_estimators" : [ i for i in range(100 , 1000 , 100 )]}
    model  = RandomForestClassifier() 
    clf  = GridSearchCV( model , param_grid , cv = 5  )
    clf.fit( X , Y )

    f.write("best params n_estimators {}".format(clf.best_params_))


    return clf.best_params_['n_estimators']

def random_forest_train_model( optparam , X_train, X_test, y_train )  :

    model  = RandomForestClassifier(n_estimators=optparam) 

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    return y_pred




def decision_trees(X_train, X_test, y_train) :  

    model = DecisionTreeClassifier() 
    model.fit(X_train , y_train)
    y_pred = model.predict(X_test )

    return y_pred




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


#writing to a file 
#file_name = "algorithm_name  Final Performance.txt"
f = open("Logistics Regression Final Performance.txt" , 'w')

for i in dfList:
    print(i)
    curcol = col[:]
    df = dfList[i]

    f.write("Subject category {} ".format(i))

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
    scores =  [  ]
    rfes = [  ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=42, test_size=0.3) 
    

    #k specifies the number of features to select  
    #conducts feature selection

    print("sdfngorhnio")
    for k in range(1, m + 1 , 1000 ):
        print( "hello" ) 
        model = LogisticRegression(max_iter=100)

        rfe = RFE(model, n_features_to_select = k).fit(X_train, y_train )

        rfes.append(rfe)
        y_pred = rfe.predict(X_test)
        scores.append(accuracy_score(y_test ,  y_pred ))


    opt_rfe  =  rfes[scores.index( max(scores))]
    arr = opt_rfe.support_

    Fcol = curcol[2:]
    feat = [  ]

    for  i in range( 0 , len(arr)) : 
        if arr[i] : 
            feat.append( Fcol[i])
    
    
    X = df.loc[:, feat].values
    X = data_filter(X, feat)

    cv = KFold(n_splits=5 )



    listacc = [ ]
    listse = [ ]
    listsp = [ ]
    foldno = 1 

    for train_index, test_index in cv.split(X):
    
        f.write("Fold Number {}\n ".format(foldno))
        foldno += 1 

        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = Y[train_index], Y[test_index]

        # optparam = random_forest_findparam(X, Y) 
        #find opt param for a specific ml algor 
        # y_pred = random_forest_train_model( optparam , X_train, X_test, y_train )


        y_pred = Logistic_reg(X_train, X_test, y_train)

        tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=['Autism', 'Control']).ravel()

    
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        se = tp / (tp + fn)
        sp = tn / (tn + fp)

        f.write("Accuracy : " + str(accuracy) + "\n" + str((tn, fp, fn, tp)) + "sp : " + str(sp) + "se : " +
                str(se)+ '\n')

        print("Accuracy : " + str(accuracy) + "\n" + str((tn, fp, fn, tp)) + "sp : " + str(sp) + "se : " +
                str(se)+ '\n')

        # print(score)

        listacc.append( accuracy )
        listsp.append(sp)
        listse.append(se)



    avgacc  = sum(listacc) / 5
    sdacc =  sum( abs( i - avgacc ) for i in listacc ) / 5 
    avgse = sum(listse) / 5 
    sdse = sum( abs( i - avgse ) for i in listse ) / 5 
    avgsp = sum(listsp) / 5  
    sdsp = sum( abs( i - avgsp ) for i in listsp ) / 5 

    f.write("accuracy : {} sd : {} \n".format(avgacc , sdacc))
    f.write("se : {} sd : {}\n ".format(avgse , sdse))
    f.write("sp : {} sd : {} \n".format(avgsp , sdsp))


    
    
    
    
    
    
