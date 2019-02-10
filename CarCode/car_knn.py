import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import os
import time
from textwrap import wrap

df = pd.read_csv("car_evaluation.csv", header=0, sep = ",", quotechar = '"')

df = df.replace('vhigh', 4)
df = df.replace('high', 3)
df = df.replace('med', 2)
df = df.replace('low', 1)
df = df.replace('5more', 6)
df = df.replace('more', 5)
df = df.replace('small', 1)
df = df.replace('big', 3)
df = df.replace('unacc', 1)
df = df.replace('acc', 2)
df = df.replace('good', 3)
df = df.replace('vgood', 4)

car = df.values
X, y = car[:,:6], car[:, 6]
X, y = X.astype(int), y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

ks = range(1, 28)

train_err = [0] * len(ks)
test_err = [0] * len(ks)
train_err2 = [0] * len(ks)
test_err2 = [0] * len(ks)
cv_scores = [0] * len(ks)
cv_scores2 = [0] * len(ks)

for i, k in enumerate(ks):
    print 'kNN: learning a kNN classifier with k = ' + str(k)
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    clf2 = KNeighborsClassifier(n_neighbors = k, weights='distance')
    clf2.fit(X_train, y_train)
    train_err[i] = accuracy_score(y_train, clf.predict(X_train))
    cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_err[i] = accuracy_score(y_test, YpredTest)
    train_err2[i] = accuracy_score(y_train, clf2.predict(X_train))
    cv_results = cross_validate(clf2, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores2[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_err2[i] = accuracy_score(y_test, YpredTest)
    print '---'

# Plot results
print 'plotting results'
plt.figure()
title = 'Car KNN: Performance vs K-Value'
plt.title('\n'.join(wrap(title,60)))
plt.plot(ks, test_err, '-', label='test unweighted')
plt.plot(ks, train_err, '-', label='train unweighted')
plt.plot(ks, cv_scores, '-', label='cross val unweighted')
plt.plot(ks, test_err2, '-', label='test weighted')
plt.plot(ks, train_err2, '-', label='train weighted')
plt.plot(ks, cv_scores2, '-', label='cross val weighted')
plt.legend()
plt.xlabel('K-Value')
plt.ylabel('Accuracy')
plt.savefig('Car_kNN.png')
print 'plot complete'
train_size = len(X_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)
cv_scores = [0] * len(offsets)
train_size = len(X_train)

for i, o in enumerate(offsets):
    print 'kNN: learning a kNN classifier with training_set_size=' + str(o)
    clf = KNeighborsClassifier(n_neighbors = 5)
    X_train_temp = X_train[:o].copy()
    y_train_temp = y_train[:o].copy()
    X_test_temp = X_test[:o].copy()
    y_test_temp = y_test[:o].copy()
    clf.fit(X_train_temp, y_train_temp)
    train_err[i] = accuracy_score(y_train_temp, clf.predict(X_train_temp))
    cv_results = cross_validate(clf, X_train_temp, y_train_temp, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test_temp)
    test_err[i] = accuracy_score(y_test_temp, YpredTest)

# Plot results
print 'plotting results'
plt.figure()
title = 'Car KNN: Performance x Training Set Size'
plt.title('\n'.join(wrap(title,60)))
plt.plot(offsets, test_err, '-', label='test')
plt.plot(offsets, train_err, '-', label='train')
plt.plot(offsets, cv_scores, '-', label='cross val')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.savefig('Car_KNN_TrainingSize.png', dpi=300)
print 'plot complete'


clf = KNeighborsClassifier(n_neighbors = 5)
t = time.time()
clf.fit(X_train, y_train)
t2 = time.time() - t
print(t2)
YpredTrain = clf.predict(X_train)
YpredTest = clf.predict(X_test)
print(confusion_matrix(y_test, YpredTest))
print(classification_report(y_test, YpredTest))
