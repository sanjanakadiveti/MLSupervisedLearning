import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from collections import Counter
import time

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

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=30)

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
f1_scores = [0] * len(depth)
accuracyTrain = [0] * len(depth)
accuracyTest = [0] * len(depth)

yscore = None

for i, d in enumerate(depth):
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
print 'plotting results'
plt.figure()
plt.title('Car Decision Tree: Accuracy vs Max Depth')
plt.plot(depth, accuracyTest, '-', label='test accuracy')
plt.plot(depth, accuracyTrain, '-', label='train accuracy')
plt.plot(depth, cv_scores, '-', label='cross val accuracy')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.savefig('Car_Dtree_Depth.png', dpi=300)
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
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
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
    clf = DecisionTreeClassifier(criterion='gini', max_depth=8)
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
    clf = DecisionTreeClassifier(criterion='gini', max_depth=11)
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
plt.title('Car Decision Tree: Accuracy Vs Training Set Size')
plt.plot(offsets, test_acc6, '-', label='test acc, max_depth = 5')
plt.plot(offsets, train_acc6, '-', label='train acc, max_depth = 5')
plt.plot(offsets, test_acc9, '-', label='test acc, max_depth = 8')
plt.plot(offsets, train_acc9, '-', label='train acc, max_depth = 8')
plt.plot(offsets, test_acc12, '-', label='test acc, max_depth = 11')
plt.plot(offsets, train_acc12, '-', label='train acc, max_depth = 11')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
filename = 'Car_Dtree_Train.png'
plt.savefig(filename, dpi=300)


data = pd.read_csv('car_evaluation.csv', sep=',', quoting = 2)['class']
c = Counter(data)
print(c)
classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 10)
t = time.time()
classifier.fit(Xtrain, Ytrain)
trainTime = time.time() - t
YpredTrain = classifier.predict(Xtrain)
YpredTest = classifier.predict(Xtest)
tree.export_graphviz(classifier, out_file='carTree.dot')
print(confusion_matrix(Ytest, YpredTest))
print('training time: ')
print(str(trainTime))
print(classification_report(Ytest, YpredTest))
