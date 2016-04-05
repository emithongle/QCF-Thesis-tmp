__author__ = 'Thong_Le'

import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier


#Lets set up a training dataset.  We'll make 100 entries, each with 19 features and
#each row classified as either 0 and 1.  We'll control the first 3 features to artificially
#set the first 3 features of rows classified as "1" to a set value, so that we know these are the "important" features.  If we do it right, the model should point out these three as important.
#The rest of the features will just be noise.
train_data = [] ##must be all floats.
for x in range(100):
    line = []
    if random.random()>0.5:
        line.append(1.0)
        #Let's add 3 features that we know indicate a row classified as "1".
        line.append(.77)
        line.append(.33)
        line.append(.55)
        for x in range(16):#fill in the rest with noise
            line.append(random.random())
    else:
        #this is a "0" row, so fill it with noise.
        line.append(0.0)
        for x in range(19):
            line.append(random.random())
    train_data.append(line)
train_data = np.array(train_data)


# Create the random forest object which will include all the parameters
# for the fit.  Make sure to set compute_importances=True
Forest = RandomForestClassifier(n_estimators = 100) #, compute_importances=True)

# Fit the training data to the training output and create the decision
# trees.  This tells the model that the first column in our data is the classification,
# and the rest of the columns are the features.
Forest = Forest.fit(train_data[0::,1::],train_data[0::,0])

#now you can see the importance of each feature in Forest.feature_importances_
# these values will all add up to one.  Let's call the "important" ones the ones that are above average.
important_features = []
for x,i in enumerate(Forest.feature_importances_):
    if i>np.average(Forest.feature_importances_):
        important_features.append(str(x))
print('Most important features:',', '.join(important_features))
#we see that the model correctly detected that the first three features are the most important, just as we expected!