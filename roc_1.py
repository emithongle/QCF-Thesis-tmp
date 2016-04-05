__author__ = 'Thong_Le'

import numpy as np

# =================================================
# 1. Mann-Whitney U Statistic
def calU(y_score):
    count = sum([1 for i in y_score[0] for j in y_score[1] if (i < j)])
    return count / (len(y_score[0]) * len(y_score[1]))

# =================================================
# 2. Approach based on the confusion matrix

def calPVUS(tp1, f12):
    return (tp1 / (tp1 + f12))

def calVUS_1(y_true, y_predicted):
    data = [[0, 0], [0, 0]]

    for i, j in zip(y_true, y_predicted):
        data[i][j] += 1

    l = [
        calPVUS(data[0][0], data[0][1]),
        calPVUS(data[1][1], data[1][0])
    ]

    return sum(l) / len(l)

# =================================================
# 3. Approach based on emperical distribution functions
#
# def calVUS_2(y_true, y_score):
#     return 0

def calPDF(data):
    return np.asarray([x / sum(data) for x in data])

def calCDF(data):
    return np.cumsum(calPDF(data))

def calVUS_2(data):
    # data = { 1 : [S_i], 2: [S_j], 3: [S_k] } for i, j, k in N

    bins = 100
    minS = min([min(data[0]), min(data[1])])
    maxS = max([max(data[0]), max(data[1])])

    count_S1, rangeS = np.histogram(np.asarray(data[0]), bins=bins, range=(minS, maxS))
    count_S2, tmp = np.histogram(np.asarray(data[1]), bins=bins, range=(minS, maxS)) #[0]

    cdf1 = calCDF(count_S1)
    pdf2 = calPDF(count_S2)

    return sum(cdf1 * pdf2)

