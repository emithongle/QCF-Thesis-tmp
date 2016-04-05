__author__ = 'Thong_Le'

import theano
import numpy as np
import theano.tensor as T
from sklearn.cross_validation import train_test_split
import store

def preprocess_1(X, y):
    return X, y

def preprocess_2(X, y):
    return X, y

def preprocess_3(X, y):
    return X, y

# Read data
def readData(name = 1):

    folder_L0 = 'data/'

    if (name == 1):
        folder_L1 = '1. Give Me Some Credit/'

    elif (name == 2):
        folder_L1 = '2. German Credit/'

    elif (name == 3):
        folder_L1 = '3. Australian Credit Approval/'


    X_0 = np.genfromtxt(folder_L0 + folder_L1 + 'data_0.csv', delimiter=',')
    y_0 = [0] * len(X_0)
    X_1 = np.genfromtxt(folder_L0 + folder_L1 + 'data_1.csv', delimiter=',')
    y_1 = [1] * len(X_1)

    X_0, y_0 = eval('preprocess_' + str(name) + '(X_0, y_0)')
    X_1, y_1 = eval('preprocess_' + str(name) + '(X_1, y_1)')

    return X_0, y_0, X_1, y_1

def randomData(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def loadData(name_id = 1):
    X_0, y_0, X_1, y_1 = readData(name_id)
    X = np.append(X_0.tolist(), X_1.tolist(), axis=0),
    y = y_0 + y_1
    return randomData(X, y)


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data(name_id = 1):
    X_0, y_0, X_1, y_1 = readData(name_id)

    X_train_0, X_valid_test_0, y_train_0, y_valid_test_0 = store.randomData(X_0, y_0, 0.4)
    X_train_1, X_valid_test_1, y_train_1, y_valid_test_1 = store.randomData(X_1, y_1, 0.4)

    X_valid_0, X_test_0, y_valid_0, y_test_0 = store.randomData(X_valid_test_0, y_valid_test_0, 0.5)
    X_valid_1, X_test_1, y_valid_1, y_test_1 = store.randomData(X_valid_test_1, y_valid_test_1, 0.5)

    train_set, valid_set, test_set = (np.append(X_train_0.tolist(), X_train_1.tolist(), axis=0), y_train_0 + y_train_1), \
                                    (np.append(X_valid_0.tolist(), X_valid_1.tolist(), axis=0), y_valid_0 + y_valid_1), \
                                    (np.append(X_test_0.tolist(), X_test_1.tolist(), axis=0), y_test_0 + y_test_1)

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval