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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

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
    
    return np.array(X), Y_color, Y_texture

def preprocess_test(filename, sep=','):
    # load data
    df = pd.read_csv(filename, sep=sep)

    X = df
    X.drop(['image', 'id', 'x', 'y', 'w', 'h'], axis=1, inplace=True)
    X = X.dropna()
    return np.array(X)

'''
The best 2 features are color histograms and HOG. Here extract those 2 set of featuress
'''
def get_first_2_features(X):
    return X[:, :315]


def KFold_train(X, y):
    balanced_accuracies = []

    for X_train, X_test, y_train, y_test in KFoldSplit(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = svm.SVC(kernel='rbf', C = 1.2, class_weight='balanced')
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        accuracy = round(balanced_accuracy_score(y_test, y_pred), 3)
        balanced_accuracies.append(accuracy)

    print('SVM Avg Balanced Accuracy: ', round(np.mean(balanced_accuracies), 3))

def KFold_train_compare(X, y):
    balanced_accuracies = []
    lr_accu = []

    for X_train, X_test, y_train, y_test in KFoldSplit(X, y):
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # X_train = get_first_2_features(X_train)
        # X_test = get_first_2_features(X_test)

        clf = svm.SVC(kernel='rbf', class_weight='balanced')
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        accuracy = round(balanced_accuracy_score(y_test, y_pred), 3)
        balanced_accuracies.append(accuracy)
        svm_pred.extend(y_pred)
        
        clf2 = GaussianNB()
        clf2.fit(X_train,y_train)
        y_pred = clf2.predict(X_test)
        accuracy = round(balanced_accuracy_score(y_test, y_pred), 3)
        lr_accu.append(accuracy)
        lr_pred.extend(y_pred)
        
        all_true.extend(y_test)
        
    print('SVM Avg Balanced Accuracy: ', round(np.mean(balanced_accuracies), 3))
    print('Gaussian Avg Balanced Accuracy: ', round(np.mean(lr_accu), 3))

def get_best_params():
    param_grid = [
    {'C': [1, 1.1, 1.2, 0.9, 0.8, 0.85], 'gamma': [0.001, 0.003, 0.002, 0.0025], 'kernel': ['rbf']},
    ]
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid=param_grid, scoring="balanced_accuracy", cv=cv)
    grid.fit(X_train, y_train)
    
    print("The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_))
    return grid.best_params_

def train(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = get_first_2_features(X_train)
    X_test = get_first_2_features(X_test)
    
    # best_params = get_best_params()
    # clf = svm.SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], class_weight='balanced')

    clf = svm.SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def output(filename, y_pred):
    y_pred = list(y_pred)

    # there is a null line so null result
    y_pred.insert(103, "null")
    y_pred = np.array(y_pred)

    with open(filename, "w") as wp:
        for pred in y_pred:
            wp.write("{}\n".format(pred))

X, Y_color, Y_texture = preprocess("data/data_train.csv")
X_test = preprocess_test("data/data_test.csv")

'''
If you want to use data_train.csv for training and testing
'''
KFold_train(X, Y_color)


'''
If you want to train the whole dataset and output its result
'''
# y_pred = train(X, X_test, Y_texture)
# output("test.csv", y_pred)

