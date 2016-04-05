__author__ = 'Thong_Le'

def get_code(tree, feature_names):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node):
                if (threshold[node] != -2):
                        print("if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        print("} else {")
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        print("}")
                else:
                        print("return " + str(value[node]))

        recurse(left, right, threshold, features, 0)


# import numpy as np
#
# from sklearn.datasets import make_blobs
# from sklearn.ensemble import RandomForestClassifier
#
# np.random.seed(0)
#
# # Generate data
# X, y = make_blobs(n_samples=1000, n_features=2, random_state=42,
#                   cluster_std=5.0)
# X_train, y_train = X[:600], y[:600]
# X_valid, y_valid = X[600:800], y[600:800]
# X_train_valid, y_train_valid = X[:800], y[:800]
# X_test, y_test = X[800:], y[800:]
#
# # Train uncalibrated random forest classifier on whole train and validation
# # data and evaluate on mytest data
# clf = RandomForestClassifier(n_estimators=25)
# clf.fit(X_train_valid, y_train_valid)
# clf_probs = clf.predict_proba(X_test)
#
# # get_code(clf.estimators_[0], )
#
# # from inspect import getmembers
# # print( getmembers( clf.estimators_[0].tree_ ) )