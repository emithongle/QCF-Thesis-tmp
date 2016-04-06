__author__ = 'Thong_Le'

import store
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.metrics import log_loss, accuracy_score
from sknn.mlp import Classifier, Layer, MultiLayerPerceptron
import numpy as np
import xlsxwriter

from itertools import combinations

from sklearn.metrics import roc_auc_score
from roc_1 import calU, calVUS_1, calVUS_2

def writeSheet(sheet, data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            sheet.write(i, j, data[i][j])

def buildClassifier(clf = 'random_forest'):
    if (clf == 'neuron_network'):
        return Classifier(
            layers=[
                Layer("Sigmoid", units=100),
                Layer("Softmax", units=2)],
            learning_rule='sgd',
            learning_rate=0.01,
            n_iter=10
        )

    elif (clf == 'SVC'):
        return svm.SVC()

    elif (clf == 'Linear Discriminant Analysis'):
        return LinearDiscriminantAnalysis()
    elif (clf == 'Quadratic Discriminant Analysis'):
        return QuadraticDiscriminantAnalysis()

    elif (clf == 'AdaBoost'):
        return AdaBoostClassifier(n_estimators=100)

    elif (clf == 'extra_trees_classifier'):
        return ExtraTreesClassifier(n_estimators=10)
    elif (clf == 'gradient_boosting_classifier'):
        return GradientBoostingClassifier(n_estimators=10)
    else:
        return RandomForestClassifier(n_estimators=10)

def getScore(type='', y_true = [], y_score = []):

    # if (type == 'ROC'):
    #     return roc_score(y_true, y_score)

    if (type == 'AUC'):
        return roc_auc_score(y_true, y_score)

    elif (type == 'U'):
        l0 = [j[0] for (i, j) in zip(y_true, y_score) if (i == 0)]
        l1 = [j[1] for (i, j) in zip(y_true, y_score) if (i == 1)]
        return calU((l0, l1))

    elif (type == 'VUS_1'):
        return calVUS_1(y_true, y_score)

    elif (type == 'VUS_2'):
        l0 = [j[0] for (i, j) in zip(y_true, y_score) if (i == 0)]
        l1 = [j[1] for (i, j) in zip(y_true, y_score) if (i == 1)]
        return calVUS_2((l0, l1))

    return None

def getFeaturesFromCombination(X, cb):
    return np.asarray([[x[i] for i in list(cb)] for x in X])

folder = 'results/'
rundate = '20160221/'

methods = {'1': 'random_forest',
           '2': 'neuron_network'
           #'3': 'SVC',
           # '4': 'Linear Discriminant Analysis',
           # '5': 'Quadratic Discriminant Analysis',
           # '6': 'AdaBoost',
           # '7': 'extra_trees_classifier',
           # '8': 'gradient_boosting_classifier'
           }

# methods = {'1': 'random_forest'}

# datas = [0, 1, 2]
datas = [0]

scores = {# '1': 'ROC',   # Recheck
          '2': 'AUC',   # Recheck
          '3': 'U',     # Recheck
          '4': 'VUS_1', # Recheck
          '5': 'VUS_2'} # Recheck

number_of_test = 10

for data_id in datas:
    print('===========================================================')
    print('- Data ', data_id + 1)
    X_0, y_0, X_1, y_1 = store.readData(data_id + 1)

    for method_id in methods:
        fd = method_id + '. ' + methods[method_id] + '/'
        print(' - Model :', methods[method_id])
        workbook = xlsxwriter.Workbook(folder + rundate + fd + str(data_id) + '.xlsx')
        data = [['#', 'Combination', 'Accuracy', 'AUC', 'U', 'VUS_1', 'VUS_2']]

        for i in range(number_of_test):
            print('   + ', methods[method_id], ' - test ', i, '...')

            X, y = np.append(X_0.tolist(), X_1.tolist(), axis=0), y_0 + y_1

            cbs = []
            for k in range(X.shape[1]):
                cbs += combinations(range(X.shape[1]), k + 1)

            for z, cb in zip(range(len(cbs)), cbs):
                print('   -> Combination ' + str(z))
                _X = getFeaturesFromCombination(X, cb)
                X_train, X_test, y_train, y_test = store.randomData(_X, y)

                clf = buildClassifier(methods[method_id])

                clf.fit(X_train, np.asarray(y_train))
                y_hat = clf.predict(X_test)
                y_score = clf.predict_proba(X_test)

                score = accuracy_score(y_test, y_hat)
                print('--> Done: ', score)
                data.append(
                    [
                        i,
                        str(list(cb)),
                        score,
                        # getScore('ROC', y_test, y_hat),
                        getScore('AUC', y_test, y_hat),
                        getScore('U', y_test, y_score),
                        getScore('VUS_1', y_test, y_hat),
                        getScore('VUS_2', y_test, y_score),
                    ]
                )

        writeSheet(workbook.add_worksheet(methods[method_id]), data)
        workbook.close()
