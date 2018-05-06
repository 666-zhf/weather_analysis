import numpy as np
from os import listdir
from os.path import isfile, join, abspath, dirname
import pandas as pd
from sklearn import neighbors
from sklearn.svm import SVR

def list_weather_files():
    relative_path = abspath(dirname(__file__))
    directory_path = join(relative_path, '..\data\\')
    return [join(directory_path, f) for f in listdir(directory_path) if isfile(join(directory_path, f))]

def load_weather_data():
    data = []

    for file_path in list_weather_files():
        file_data = pd.read_csv(file_path, sep=" ").as_matrix()
        if(len(data) == 0):
            data = file_data
        else:
            data = np.concatenate((data, file_data))

    return data

def knn_regression_min_temp(data):
    train_data = data[0:1000, :]
    test_data = data[1000:len(data) - 1, :]

    train_X = train_data[:, [0,2,3,4,5,6]]
    train_y = train_data[:,1]
    test_X = test_data[:, [0,2,3,4,5,6]]
    test_y = test_data[:,1]
    
    n_neighbors = 6
    knn = neighbors.KNeighborsRegressor(n_neighbors, p=2, weights='uniform')
    print knn.fit(train_X, train_y).score(test_X, test_y)

def svm_regression_min_temp(data):
    train_data = data[0:1000, :]
    test_data = data[1000:len(data) - 1, :]

    train_X = train_data[:, [0,2,3,4,5,6]]
    train_y = train_data[:,1]
    test_X = test_data[:, [0,2,3,4,5,6]]
    test_y = test_data[:,1]
    
    clf = SVR(C=1.0, epsilon=0.2, gamma=0.01)
    print clf.fit(train_X, train_y).score(test_X, test_y)

def knn_regression_max_temp(data):
    train_data = data[0:1000, :]
    test_data = data[1000:len(data) - 1, :]

    train_X = train_data[:, 1:6]
    train_y = train_data[:,0]
    test_X = test_data[:, 1:6]
    test_y = test_data[:,0]

    n_neighbors = 6
    knn = neighbors.KNeighborsRegressor(n_neighbors, p=2, weights='uniform')
    print knn.fit(train_X, train_y).score(test_X, test_y)

def svm_regression_max_temp(data):
    train_data = data[0:1000, :]
    test_data = data[1000:len(data) - 1, :]

    train_X = train_data[:, 1:6]
    train_y = train_data[:,0]
    test_X = test_data[:, 1:6]
    test_y = test_data[:,0]

    clf = SVR(C=1.0, epsilon=0.2, gamma=0.01)
    print clf.fit(train_X, train_y).score(test_X, test_y)

weather_data = load_weather_data()

knn_regression_min_temp(weather_data)
svm_regression_min_temp(weather_data)
knn_regression_max_temp(weather_data)
svm_regression_max_temp(weather_data)