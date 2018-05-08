import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join, abspath, dirname
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def list_weather_files():
    relative_path = abspath(dirname(__file__))
    directory_path = join(relative_path, '..\data\\')
    files = [join(directory_path, f) for f in listdir(directory_path) if isfile(join(directory_path, f))]
    files = sorted(files, key=lambda f: int(f.split('-')[-1].split('.')[0]), reverse=1)
    result = list()

    for f in files:
        result.append((f.split('-')[-1].split('.')[0], f))
    
    return result

def load_weather_data():
    data = []

    paths = list_weather_files()

    for file_path in paths:
        file_data = pd.read_csv(file_path[1], sep=" ").as_matrix()

        if(len(data) == 0):
            data = file_data
        else:            
            data = np.concatenate((data, file_data))
            
    return data

def split_features_label(data):
    data_X = data[:,[0, 1, 2, 3, 5, 6]]
    labels = data[:, 4]
    data_y = []
    for label in labels:
        if(label > 0):
            data_y.append(1)
        else:
            data_y.append(0)
            
    return data_X, data_y

def knn_classification(data, train_test_percentage=20):
    data_X, data_y = split_features_label(data)
    test_size = float(train_test_percentage) / 100
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=test_size, random_state=0)
    
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_X, train_y)
    pred_y = neigh.predict(test_X)
    print 'KNN classification'
    print 'Score: '+ str(accuracy_score(test_y, pred_y))
    print(classification_report(test_y, pred_y))

def logistic_classification(data, train_test_percentage=20):
    data_X, data_y = split_features_label(data)
    test_size = float(train_test_percentage) / 100
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=test_size, random_state=0)

    logisticRegr = LogisticRegression(penalty='l1')
    logisticRegr.fit(train_X, train_y)
    pred_y = logisticRegr.predict(test_X)
    print 'Logistic classification'
    print 'Score: '+ str(accuracy_score(test_y, pred_y))
    print(classification_report(test_y, pred_y))

def svm_classification(data, train_test_percentage=20):
    data_X, data_y = split_features_label(data)
    test_size = float(train_test_percentage) / 100
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=test_size, random_state=0)

    clf = SVC()
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    print 'SVM classification'
    print 'Score: '+ str(accuracy_score(test_y, pred_y))
    print(classification_report(test_y, pred_y))

train_test_percentage = 20
weather_data = load_weather_data()
knn_classification(weather_data, train_test_percentage)
logistic_classification(weather_data, train_test_percentage)
svm_classification(weather_data, train_test_percentage)