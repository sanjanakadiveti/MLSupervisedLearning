import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import svm
import os
import time
from textwrap import wrap

df = pd.read_csv("dataset.csv", sep = ",", quotechar = '"')
df = pd.get_dummies(df)
X = df.ix[:, df.columns != 'Result']
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#
#
clf = svm.SVC(C=1, degree=1, kernel='poly')
clf.fit(X_train, y_train)
print("poly 1")
print(str(accuracy_score(y_train, clf.predict(X_train))))
print(str(accuracy_score(y_test, clf.predict(X_test))))

clf = svm.SVC(C=1, degree=2, kernel='poly')
clf.fit(X_train, y_train)
print("poly 2")
print(str(accuracy_score(y_train, clf.predict(X_train))))
print(str(accuracy_score(y_test, clf.predict(X_test))))

clf = svm.SVC(C=1, degree=3, kernel='poly')
clf.fit(X_train, y_train)
print("poly 3")
print(str(accuracy_score(y_train, clf.predict(X_train))))
print(str(accuracy_score(y_test, clf.predict(X_test))))

clf2 = svm.SVC(C=1, degree=3, kernel='rbf')
clf2.fit(X_train, y_train)
print('rbf')
print(str(accuracy_score(y_train, clf2.predict(X_train))))
print(str(accuracy_score(y_test, clf2.predict(X_test))))

train_size = len(X_train)
offsets = range(int(-10), int(4), int(2))

train_errP = [0] * len(offsets)
test_errP = [0] * len(offsets)
train_err2 = [0] * len(offsets)
test_err2 = [0] * len(offsets)
train_errP2 = [0] * len(offsets)
test_errP2 = [0] * len(offsets)
train_errP3 = [0] * len(offsets)
test_errP3 = [0] * len(offsets)

train_size = len(X_train)

for i, o in enumerate(offsets):
    print(o)
    clf = svm.SVC(C=1, degree=1, gamma=pow(2,o), kernel='poly')
    clf.fit(X_train, y_train)
    train_errP[i] = accuracy_score(y_train, clf.predict(X_train))
    cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_errP[i] = accuracy_score(y_test, YpredTest)
    print '---'
    clf = svm.SVC(C=1, degree=2, gamma=pow(2,o), kernel='poly')
    clf.fit(X_train, y_train)
    train_errP2[i] = accuracy_score(y_train, clf.predict(X_train))
    cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_errP2[i] = accuracy_score(y_test, YpredTest)
    print '---'
    clf = svm.SVC(C=1, degree=3, gamma=pow(2,o), kernel='poly')
    clf.fit(X_train, y_train)
    train_errP3[i] = accuracy_score(y_train, clf.predict(X_train))
    cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_errP3[i] = accuracy_score(y_test, YpredTest)
    print '---'
    clf2 = svm.SVC(C=1, degree=3, gamma=pow(2,o), kernel='rbf')
    clf2.fit(X_train, y_train)
    train_err2[i] = accuracy_score(y_train, clf2.predict(X_train))
    cv_results = cross_validate(clf2, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_err2[i] = accuracy_score(y_test, YpredTest)

    # Plot results
print 'plotting results'
plt.figure()
title = 'Phishing SVM Performance vs Gamma'
plt.title('\n'.join(wrap(title,60)))
plt.plot(offsets, test_errP, '-', label='test poly degree = 1')
plt.plot(offsets, train_errP, '-', label='train poly degree = 1')
plt.plot(offsets, test_errP2, '-', label='test poly degree = 2')
plt.plot(offsets, train_errP2, '-', label='train poly degree = 2')
plt.plot(offsets, test_errP3, '-', label='test poly degree = 3')
plt.plot(offsets, train_errP3, '-', label='train poly degree = 3')
plt.plot(offsets, test_err2, '-', label='test rbf')
plt.plot(offsets, train_err2, '-', label='train rbf')
plt.legend()
plt.xlabel('Log2 Gamma')
plt.ylabel('Accuracy')
plt.savefig('Phishing_SVM_Gamma.png')
print 'plot complete'


train_size = len(X_train)
offsets = range(int(-10), int(10), int(2))
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)
train_err2 = [0] * len(offsets)
test_err2 = [0] * len(offsets)
cv_scores = [0] *len(offsets)
train_size = len(X_train)

for i, o in enumerate(offsets):
    print '---'
    # clf = svm.SVC(C=pow(2,o), degree=3, gamma=0.0625, kernel='linear')
    # clf.fit(X_train, y_train)
    # train_err[i] = accuracy_score(y_train, clf.predict(X_train))
    # test_err[i] = accuracy_score(y_test, clf.predict(X_test))
    clf2 = svm.SVC(C=pow(2,o), degree=3, gamma=0.0625, kernel='rbf')
    clf2.fit(X_train, y_train)
    train_err2[i] = accuracy_score(y_train, clf2.predict(X_train))
    cv_results = cross_validate(clf2, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_err2[i] = accuracy_score(y_test, YpredTest)

    # Plot results
print 'plotting results'
plt.figure()
title = 'Phishing SVM Performance vs C'
plt.title('\n'.join(wrap(title,60)))
# plt.plot(offsets, test_err, '-', label='test linear')
# plt.plot(offsets, train_err, '-', label='train linear')
plt.plot(offsets, test_err2, '-', label='test rbf')
plt.plot(offsets, train_err2, '-', label='train rbf')
plt.plot(offsets, cv_scores, '-', label='cross val')
plt.legend()
plt.xlabel('Log2 C')
plt.ylabel('Accuracy')
plt.savefig('Phishing_SVM_C.png')
print 'plot complete'


train_size = len(X_train)
offsets = range(int(0.1 * train_size), int(train_size), int(200))
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)
cv_scores = [0] *len(offsets)
for i, o in enumerate(offsets):
    print(o)
    layers = []
    for _ in range(5): layers.append(21)
    X_train_temp = X_train[:o].copy()
    y_train_temp = y_train[:o].copy()
    X_test_temp = X_test[:o].copy()
    y_test_temp = y_test[:o].copy()
    clf = svm.SVC(C=1, degree=3, gamma=0.0625, kernel='rbf')
    clf.fit(X_train_temp, y_train_temp)
    train_err[i] = accuracy_score(y_train_temp, clf.predict(X_train_temp))
    cv_results = cross_validate(clf, X_train_temp, y_train_temp, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test_temp)
    test_err[i] = accuracy_score(y_test_temp, YpredTest)
plt.figure()
plt.plot(offsets, test_err, '-', label='test')
plt.plot(offsets, train_err, '-', label='train')
plt.plot(offsets, cv_scores, '-', label='cross val')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
title = 'Phishing SVM: Accuracy vs Training Size'
plt.title('\n'.join(wrap(title,60)))
filename = 'Phishing_SVM_TrainSize.png'
plt.savefig(filename, dpi=300)
print 'plot complete'
print 'training_set_max_size:', train_size, '\n'

clf = svm.SVC(C=1, degree=3, gamma=0.0625, kernel='rbf')
t = time.time()
clf.fit(X_train, y_train)
print(time.time() - t)
YpredTrain = clf.predict(X_train)
YpredTest = clf.predict(X_test)
print(confusion_matrix(y_test, YpredTest))
print(classification_report(y_test, YpredTest))
