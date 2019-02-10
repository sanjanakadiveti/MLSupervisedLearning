import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from sklearn import tree
from collections import Counter
import time

df = pd.read_csv("dataset.csv", sep = ",", quotechar = '"')
df = pd.get_dummies(df)
X = df.ix[:, df.columns != 'Result']
Y = df['Result']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.30, random_state = 30)

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

depth = range(2, 21)
trainErr = [0] * len(depth)
testErr = [0] * len(depth)
trainTimes = [0] * len(depth)
predTrainTimes = [0] * len(depth)
predTestTimes = [0] * len(depth)
cv_scores = [0] * len(depth)
accuracyTrain = [0] * len(depth)
accuracyTest = [0] * len(depth)
f1_scores = [0] * len(depth)

for i, d in enumerate(depth):
    print(i)
    classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = d)
    t = time.time()
    classifier.fit(Xtrain, Ytrain)
    trainTimes[i] = time.time() - t
    YpredTrain = classifier.predict(Xtrain)
    accuracyTrain[i] = accuracy_score(Ytrain, YpredTrain)
    cv_results = cross_validate(classifier, Xtrain, Ytrain, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(Xtest)
    accuracyTest[i] = accuracy_score(Ytest, YpredTest)
    # f1_scores[i] = f1_score(Ytest, YpredTest, average= 'macro')

print 'plotting results'
plt.figure()
plt.title('Phishing Decision Tree: Accuracy vs Max Depth')
plt.plot(depth, accuracyTest, '-', label='test accuracy')
plt.plot(depth, accuracyTrain, '-', label='train accuracy')
plt.plot(depth, cv_scores, '-', label='cross val accuracy')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.savefig('Phishing_Dtree_Depth.png', dpi=300)

print('training times:')
print(trainTimes)

train_size = len(Xtrain)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
train_acc6 = [0] * len(offsets)
test_acc6 = [0] * len(offsets)
train_acc9 = [0] * len(offsets)
test_acc9 = [0] * len(offsets)
train_acc12 = [0] * len(offsets)
test_acc12 = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'
for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=6)
    X_train_temp = Xtrain[:o].copy()
    y_train_temp = Ytrain[:o].copy()
    X_test_temp = Xtest[:o].copy()
    y_test_temp = Ytest[:o].copy()
    clf = clf.fit(X_train_temp, y_train_temp)

    train_acc6[i] = accuracy_score(y_train_temp, clf.predict(X_train_temp))
    cv_results = cross_validate(clf, X_train_temp, y_train_temp, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test_temp)
    test_acc6[i] = accuracy_score(y_test_temp, YpredTest)

for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=9)
    X_train_temp = Xtrain[:o].copy()
    y_train_temp = Ytrain[:o].copy()
    X_test_temp = Xtest[:o].copy()
    y_test_temp = Ytest[:o].copy()
    clf = clf.fit(X_train_temp, y_train_temp)

    train_acc9[i] = accuracy_score(y_train_temp, clf.predict(X_train_temp))
    cv_results = cross_validate(clf, X_train_temp, y_train_temp, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test_temp)
    test_acc9[i] = accuracy_score(y_test_temp, YpredTest)

for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=12)
    X_train_temp = Xtrain[:o].copy()
    y_train_temp = Ytrain[:o].copy()
    X_test_temp = Xtest[:o].copy()
    y_test_temp = Ytest[:o].copy()
    clf = clf.fit(X_train_temp, y_train_temp)

    train_acc12[i] = accuracy_score(y_train_temp,clf.predict(X_train_temp))
    cv_results = cross_validate(clf, X_train_temp, y_train_temp, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test_temp)
    test_acc12[i] = accuracy_score(y_test_temp, YpredTest)

print 'plotting results'
plt.figure()
plt.title('Phishing Decision Tree: Accuracy Vs Training Set Size for Max Depths')
plt.plot(offsets, test_acc6, '-', label='test acc, max_depth = 6')
plt.plot(offsets, train_acc6, '-', label='train acc, max_depth = 6')
plt.plot(offsets, test_acc9, '-', label='test acc, max_depth = 9')
plt.plot(offsets, train_acc9, '-', label='train acc, max_depth = 9')
plt.plot(offsets, test_acc12, '-', label='test acc, max_depth = 12')
plt.plot(offsets, train_acc12, '-', label='train acc, max_depth = 12')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
filename = 'Phishing_Dtree_AccVsTrain.png'
plt.savefig(filename, dpi=300)


data = pd.read_csv('dataset.csv', sep=',', quoting = 2)['Result']
c = Counter(data)
print('Counter of Labels')
print('--------------')
# print('1.0: ' + str(c[1]))
# print('-1.0: ' + str(c[-1]))
# print(c)
#
# # standard classifier
# # max depth of 13
classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 9)
t = time.time()
classifier.fit(Xtrain, Ytrain)
trainTime = time.time() - t
print(trainTime)
# YpredTrain = classifier.predict(Xtrain)
# YpredTest = classifier.predict(Xtest)
# tree.export_graphviz(classifier, out_file='tree.dot')
# print(confusion_matrix(Ytest, YpredTest))
# print('training time: ')
# print(str(trainTime))
# print(classification_report(Ytest, YpredTest))
