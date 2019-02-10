import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
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

#NNClassifier
train_size = len(X_train)
r = range(1,35)
train_err = [0] * len(r)
test_err = [0] * len(r)
cv_scores = [0] * len(r)
for n in r:
    mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, epsilon=1e-08,
       hidden_layer_sizes=(n, ), learning_rate='constant',
       learning_rate_init=0.001, max_iter=600, momentum=0.9,
       nesterovs_momentum=True, random_state=1,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
    mlp.fit(X_train,y_train)
    print(n)
    train_err[n-1] = accuracy_score(y_train, mlp.predict(X_train))
    cv_results = cross_validate(mlp, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[n-1] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_err[n-1] = accuracy_score(y_test, YpredTest)
    # test_err[n-1] = accuracy_score(y_test, mlp.predict(X_test))
plt.plot(r, test_err, '-', label='test')
plt.plot(r, train_err, '-', label='train')
plt.plot(r, cv_scores, '-', label='cross val')
plt.legend()
plt.xlabel('Neurons')
plt.ylabel('Accuracy')
title = 'Car NN: Accuracy vs Neurons'
plt.title('\n'.join(wrap(title,60)))
filename = 'Car_NN_tanh_Neurons.png'
plt.savefig(filename, dpi=300)
print 'plot complete'

r = range(1,20)
train_err = [0] * len(r)
test_err = [0] * len(r)
cv_scores = [0] * len(r)
for l in r:
    print(l)
    layers = []
    for _ in range(l): layers.append(21)
    mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(layers), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,nesterovs_momentum=True, random_state=1,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
    mlp.fit(X_train,y_train)
    train_err[l-1] = accuracy_score(y_train, mlp.predict(X_train))
    cv_results = cross_validate(mlp, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[l-1] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_err[l-1] = accuracy_score(y_test, YpredTest)

    # test_err[l-1] = accuracy_score(y_test, mlp.predict(X_test))
plt.figure()
plt.plot(r, test_err, '-', label='test')
plt.plot(r, train_err, '-', label='train')
plt.plot(r, cv_scores, '-', label='cross val')
plt.legend()
plt.xlabel('Layers')
plt.ylabel('Accuracy')
title = 'Car NN: Performance vs Layers'
plt.title('\n'.join(wrap(title,60)))
filename = 'Car_NN_Layers.png'
plt.savefig(filename, dpi=300)
print 'plot complete'
print 'training_set_max_size:', train_size, '\n'

offsets = range(int(0.1 * 600), int(600), int(20))
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)
cv_scores = [0] * len(offsets)
for i, o in enumerate(offsets):
    print(o)
    layers = []
    for _ in range(5): layers.append(21)
    mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(layers), learning_rate='constant',
       learning_rate_init=0.001, max_iter=o, momentum=0.9,nesterovs_momentum=True, random_state=1,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
    mlp.fit(X_train,y_train)
    train_err[i] = accuracy_score(y_train, mlp.predict(X_train))
    cv_results = cross_validate(mlp, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test)
    test_err[i] = accuracy_score(y_test, YpredTest)
    # test_err[i] = accuracy_score(y_test, mlp.predict(X_test))
plt.figure()
plt.plot(offsets, test_err, '-', label='test')
plt.plot(offsets, train_err, '-', label='train')
plt.plot(offsets, cv_scores, '-', label='cross val')
plt.legend()
plt.xlabel('Max Iterations')
plt.ylabel('Accuracy')
title = 'Car NN: Accuracy vs Max Iterations'
plt.title('\n'.join(wrap(title,60)))
filename = 'Car_NN_MaxIter.png'
plt.savefig(filename, dpi=300)
print 'plot complete'
print 'training_set_max_size:', train_size, '\n'

train_size = len(X_train)
offsets = range(int(0.1 * train_size), int(train_size), int(0.1 * train_size))
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)
cv_scores = [0] * len(offsets)
for i, o in enumerate(offsets):
    print(o)
    layers = []
    for _ in range(5): layers.append(21)
    X_train_temp = X_train[:o].copy()
    y_train_temp = y_train[:o].copy()
    X_test_temp = X_test[:o].copy()
    y_test_temp = y_test[:o].copy()
    mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(layers), learning_rate='constant',
       learning_rate_init=0.001, max_iter=220, momentum=0.9,nesterovs_momentum=True, random_state=1,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
    mlp.fit(X_train_temp, y_train_temp)
    train_err[i] = accuracy_score(y_train_temp, mlp.predict(X_train_temp))
    cv_results = cross_validate(mlp, X_train_temp, y_train_temp, cv=5, scoring='accuracy', return_train_score=True, return_estimator=True)
    cv_scores[i] = cv_results['test_score'].mean()
    index = np.argmax(cv_results['test_score'])
    estimator = cv_results['estimator'][index]
    YpredTest = estimator.predict(X_test_temp)
    test_err[i] = accuracy_score(y_test_temp, YpredTest)
    # test_err[i] = accuracy_score(y_test_temp, mlp.predict(X_test_temp))
plt.figure()
plt.plot(offsets, test_err, '-', label='test')
plt.plot(offsets, train_err, '-', label='train')
plt.plot(offsets, cv_scores, '-', label='cross val')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
title = 'Car NN: Accuracy vs Training Size'
plt.title('\n'.join(wrap(title,60)))
filename = 'Car_NN_TrainSize.png'
plt.savefig(filename, dpi=300)
print 'plot complete'
print 'training_set_max_size:', train_size, '\n'


layers = []
for _ in range(5): layers.append(21)
clf = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
   beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(layers), learning_rate='constant',
   learning_rate_init=0.001, max_iter=220, momentum=0.9,nesterovs_momentum=True, random_state=1,
   shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
t = time.time()
clf = clf.fit(X_train, y_train)
t1 = time.time() - t
print(t1)
YpredTrain = clf.predict(X_train)
t2 = time.time() - t1
YpredTest = clf.predict(X_test)
print(confusion_matrix(y_test, YpredTest))
print(classification_report(y_test, YpredTest))
