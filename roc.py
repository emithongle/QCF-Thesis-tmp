__author__ = 'Thong_Le'

import numpy as np

# =================================================
# 1. Mann-Whitney U Statistic
def calU(data):
    # data = { 1 : [S_i], 2: [S_j], 3: [S_k] } for i, j, k in N

    count = 0
    for i in data[1]:
        for j in data[2]:
            for k in data[3]:
                if ((i < j) and (j < k)):
                    count += 1

    return count / (len(data[1]) * len(data[2]) * len(data[3]))


# =================================================
# 2. Approach based on the confusion matrix
def calPVUS(tp1, tp2, f12, f13, f23):
    return (tp1 / (tp1 + f12 + f13)) * (tp2 / (tp2 + f23))

def calVUS_1(data):
    # data = list(M_3x3)

    l = [
        calPVUS(data['tp1'], data['tp2'], data['f12'], data['f13'], data['f23']),
        calPVUS(data['tp1'], data['tp3'], data['f13'], data['f12'], data['f32']),

        calPVUS(data['tp2'], data['tp1'], data['f21'], data['f23'], data['f13']),
        calPVUS(data['tp2'], data['tp3'], data['f23'], data['f21'], data['f31']),

        calPVUS(data['tp3'], data['tp1'], data['f31'], data['f32'], data['f12']),
        calPVUS(data['tp3'], data['tp2'], data['f32'], data['f31'], data['f21']),
    ]

    return sum(l) / len(l)

# =================================================
# 3. Approach based on emperical distribution functions
def calPDF(data):
    return [x / sum(data) for x in data]

def calCDF(data):
    return list(np.cumsum(calPDF(data)))

def calVUS_2(data):
    # data = { 1 : [S_i], 2: [S_j], 3: [S_k] } for i, j, k in N

    bins = 100
    minS = min([min(data[1]), min(data[2]), min(data[3])])
    maxS = max([max(data[1]), max(data[2]), max(data[3])])

    count_S1, rangeS = np.histogram(np.asarray(data[1]), bins=bins, range=(minS, maxS))
    count_S2 = np.histogram(np.asarray(data[2]), bins=bins, range=(minS, maxS))[0]
    count_S3 = np.histogram(np.asarray(data[3]), bins=bins, range=(minS, maxS))[0]

    cdf1 = calCDF(count_S1)
    pdf2 = calPDF(count_S2)
    cdf3 = calCDF(count_S3)

    return sum(cdf1 * (1 - cdf3) * pdf2)

