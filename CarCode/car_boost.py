import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 30)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

train_size = len(X_train)
max_n_estimators = range(2, 31, 1)
train_acc4 = [0] * len(max_n_estimators)
test_acc4 = [0] * len(max_n_estimators)
train_acc6 = [0] * len(max_n_estimators)
test_acc6 = [0] * len(max_n_estimators)
train_acc8 = [0] * len(max_n_estimators)
test_acc8 = [0] * len(max_n_estimators)
train_acc10 = [0] * len(max_n_estimators)
test_acc10 = [0] * len(max_n_estimators)
train_acc12 = [0] * len(max_n_estimators)
test_acc12 = [0] * len(max_n_estimators)

cv_scores4 = [0] * len(max_n_estimators)
cv_scores6 = [0] * len(max_n_estimators)
cv_scores8 = [0] * len(max_n_estimators)
cv_scores10 = [0] * len(max_n_estimators)
cv_scores12 = [0] * len(max_n_estimators)

for i, o in enumerate(max_n_estimators):
    dt4 = DecisionTreeClassifier(max_depth=2)
    dt6 = DecisionTreeClassifier(max_depth=4)
    dt8 = DecisionTreeClassifier(max_depth=6)
    dt10 = DecisionTreeClassifier(max_depth=8)
    dt12 = DecisionTreeClassifier(max_depth=10)
    bdt4 = AdaBoostClassifier(base_estimator=dt4, n_estimators=o)
    bdt6 = AdaBoostClassifier(base_estimator=dt6, n_estimators=o)
    bdt8 = AdaBoostClassifier(base_estimator=dt8, n_estimators=o)
    bdt10 = AdaBoostClassifier(base_estimator=dt10, n_estimators=o)
    bdt12 = AdaBoostClassifier(base_estimator=dt12, n_estimators=o)
    bdt4.fit(X_train, y_train)
    bdt6.fit(X_train, y_train)
    bdt8.fit(X_train, y_train)
    bdt10.fit(X_train, y_train)
    bdt12.fit(X_train, y_train)
    train_acc4[i] = accuracy_score(y_train, bdt4.predict(X_train))
    cv_results = cross_validate(bdt4, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    # cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_acc4[i] = accuracy_score(y_test, YpredTest)
    # test_acc4[i] = accuracy_score(y_test, bdt4.predict(X_test))

    # cv_scores4[i] = (cross_val_score(bdt4, X_train, y_train, cv=10, scoring='accuracy').mean())
    train_acc6[i] = accuracy_score(y_train, bdt6.predict(X_train))
    cv_results = cross_validate(bdt6, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    # cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_acc6[i] = accuracy_score(y_test, YpredTest)
    # test_acc6[i] = accuracy_score(y_test, bdt6.predict(X_test))
    # cv_scores6[i] = (cross_val_score(bdt6, X_train, y_train, cv=10, scoring='accuracy').mean())
    train_acc8[i] = accuracy_score(y_train, bdt8.predict(X_train))
    cv_results = cross_validate(bdt8, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    # cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_acc8[i] = accuracy_score(y_test, YpredTest)
    # test_acc8[i] = accuracy_score(y_test, bdt8.predict(X_test))
    # cv_scores8[i] = (cross_val_score(bdt8, X_train, y_train, cv=10, scoring='accuracy').mean())
    train_acc10[i] = accuracy_score(y_train, bdt10.predict(X_train))
    cv_results = cross_validate(bdt10, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    # cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_acc10[i] = accuracy_score(y_test, YpredTest)
    # test_acc10[i] = accuracy_score(y_test, bdt10.predict(X_test))
    # cv_scores10[i] = (cross_val_score(bdt10, X_train, y_train, cv=10, scoring='accuracy').mean())
    train_acc12[i] = accuracy_score(y_train, bdt12.predict(X_train))
    cv_results = cross_validate(bdt12, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    # cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_acc12[i] = accuracy_score(y_test, YpredTest)
    # test_acc12[i] = accuracy_score(y_test, bdt12.predict(X_test))
    # cv_scores12[i] = (cross_val_score(bdt12, X_train, y_train, cv=10, scoring='accuracy').mean())

# Plot results
print 'plotting results'
plt.figure()
title = 'Car AdaBoost Decision Tree: Accuracy x Num Estimators'
plt.title('\n'.join(wrap(title,60)))
plt.plot(max_n_estimators, test_acc4, '-', label='test acc, max_depth = 2')
plt.plot(max_n_estimators, train_acc4, '-', label='train acc, max_depth = 2')
# plt.plot(max_n_estimators, cv_scores4, '-', label='cross val acc, max_depth = 4')
plt.plot(max_n_estimators, test_acc6, '-', label='test acc, max_depth = 4')
plt.plot(max_n_estimators, train_acc6, '-', label='train acc, max_depth = 4')
# plt.plot(max_n_estimators, cv_scores6, '-', label='cross val acc, max_depth = 6')
plt.plot(max_n_estimators, test_acc8, '-', label='test acc, max_depth = 6')
plt.plot(max_n_estimators, train_acc8, '-', label='train acc, max_depth = 6')
# plt.plot(max_n_estimators, cv_scores8, '-', label='cross val acc, max_depth = 8')
plt.plot(max_n_estimators, test_acc10, '-', label='test acc, max_depth = 8')
plt.plot(max_n_estimators, train_acc10, '-', label='train acc, max_depth = 8')
# plt.plot(max_n_estimators, cv_scores10, '-', label='cross val acc, max_depth = 10')
plt.plot(max_n_estimators, test_acc12, '-', label='test acc, max_depth = 10')
plt.plot(max_n_estimators, train_acc12, '-', label='train acc, max_depth = 10')
# plt.plot(max_n_estimators, cv_scores12, '-', label='cross val acc, max_depth = 12')
plt.legend(loc=2, prop={'size': 6})
plt.xlabel('Num Estimators')
plt.ylabel('Accuracy')
plt.savefig('Car_Boost_NumEstimators.png', dpi=300)
print 'plot complete'

offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
MAX_DEPTH = 6
NUM_EST = 10
train_acc = [0] * len(offsets)
test_acc = [0] * len(offsets)
cv_scores = [0] * len(offsets)

for i, o in enumerate(offsets):
    X_train_temp = X_train[:o].copy()
    y_train_temp = y_train[:o].copy()
    X_test_temp = X_test[:o].copy()
    y_test_temp = y_test[:o].copy()
    t = DecisionTreeClassifier(max_depth = MAX_DEPTH)
    clf = AdaBoostClassifier(base_estimator=t, n_estimators=NUM_EST)
    clf = clf.fit(X_train_temp, y_train_temp)
    train_acc[i] = accuracy_score(y_train_temp, clf.predict(X_train_temp))
    cv_results = cross_validate(clf, X_train_temp, y_train_temp, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test_temp)
    test_acc[i] = accuracy_score(y_test_temp, YpredTest)
    # test_acc[i] = accuracy_score(y_test_temp, clf.predict(X_test_temp))
print 'plotting results'
plt.figure()
plt.title('Car Boosted Decision Tree: Accuracy x Training Set Size')
plt.plot(offsets, test_acc, '-', label='test')
plt.plot(offsets, train_acc, '-', label='train')
plt.plot(offsets, cv_scores, '-', label='cross val')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.savefig('Car_Boost_TrainSize.png', dpi=300)
### ---

t = DecisionTreeClassifier(max_depth = MAX_DEPTH)
clf = AdaBoostClassifier(base_estimator=t, n_estimators=NUM_EST)
t = time.time()
clf = clf.fit(X_train, y_train)
t2 = time.time() - t
print(t2)
YpredTrain = clf.predict(X_train)
YpredTest = clf.predict(X_test)
print(confusion_matrix(y_test, YpredTest))
print(classification_report(y_test, YpredTest))
