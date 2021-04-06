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

def one_hot(Y, prefix):
    one_hot = pd.get_dummies(Y)
    names = []
    for col in one_hot.columns:
        names.append(prefix + '_' + col)
    one_hot.columns = names
    return one_hot
        
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
    Y_color = one_hot(Y_color, "color")
    # Y_texture = one_hot(Y_texture, "texture")
    
    # return X, Y_color, Y_texture

def train_color(X, y):
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

preprocess("data/data_train.csv")