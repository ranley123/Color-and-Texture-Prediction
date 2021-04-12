import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm

def KFoldSplit(X, y):
    kf = KFold()
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]

        yield X_train, X_test, y_train, y_test
        
def preprocess(filename, sep=','):
    # load data
    df = pd.read_csv(filename, sep=sep)
    df = df.dropna()
    # shuffle
    df = shuffle(df)

    X = df
    Y_color = df.iloc[:, 6]
    Y_texture = df.iloc[:, 7]
    X.drop(['color', 'texture', 'image', 'id', 'x', 'y', 'w', 'h'], axis=1, inplace=True)
    
    
#     encoding y
    Y_color = np.array(Y_color)
    Y_texture = np.array(Y_texture)
    
#     print(X.head())
    
    return X, Y_color, Y_texture

def eval(y_test, y_pred):
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    balanced_accu = round(balanced_accuracy_score(y_test, y_pred), 3)
    
    print('accuracy: ', accuracy)
    print('balanced accuracy: ', balanced_accu)

    return accuracy, balanced_accu

def KFold_train(X, y):
    accuracies = []
    balanced_accuracies = []

    for X_train, X_test, y_train, y_test in KFoldSplit(X, y):
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
        
        clf = svm.SVC(kernel='rbf', class_weight='balanced')
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        accuracy, balanced = eval(y_test, y_pred)
        accuracies.append(accuracy)
#         balanced_accuracies.append(balanced)
    print('Avg Accuracy: ', round(np.mean(accuracies), 3))
#     print('Avg Balanced Accuracy: ', round(np.mean(balanced_accuracies), 3))

X, Y_color, Y_texture = preprocess("data/data_train.csv")