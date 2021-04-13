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
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

def KFoldSplit(X, y):
    kf = KFold()
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]

        yield np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
        
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
    
    return np.array(X), Y_color, Y_texture

def preprocess_test(filename, sep=','):
    # load data
    df = pd.read_csv(filename, sep=sep)

    X = df
    X.drop(['image', 'id', 'x', 'y', 'w', 'h'], axis=1, inplace=True)
#     print(X.isna().any(axis=1).to_string())
    X = X.dropna()
    return np.array(X)

def KFold_train(X, y):
    balanced_accuracies = []

    for X_train, X_test, y_train, y_test in KFoldSplit(X, y):
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        pca = PCA(n_components='mle', svd_solver = 'full')# adjust yourself
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        
#         X_train = X_train[:, 27:]
#         X_test = X_test[:, 27:]
#         C_range = np.logspace(-2, 10, 13)
#         gamma_range = np.logspace(-9, 3, 13)
#         param_grid = dict(gamma=gamma_range, C=C_range)
#         cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#         grid = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid=param_grid, scoring="balanced_accuracy", cv=cv)
#         grid.fit(X_train, y_train)
        
#         print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
        clf = svm.SVC(kernel='poly', degree=3, class_weight='balanced')
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        accuracy = balanced_accu = round(balanced_accuracy_score(y_test, y_pred), 3)
        print("Balanced Accuracy: ", accuracy)
        balanced_accuracies.append(accuracy)
    print('Avg Balanced Accuracy: ', round(np.mean(balanced_accuracies), 3))

def train(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = svm.SVC(kernel='rbf',class_weight='balanced')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def output(filename, y_pred):
    y_pred = np.array(y_pred)
    with open(filename, "w") as wp:
        for pred in y_pred:
            wp.write("{}\n".format(pred))

X, Y_color, Y_texture = preprocess("data/data_train.csv")
X_test = preprocess_test("data/data_test.csv")

Y_color_pred = KFold_train(X, Y_color)