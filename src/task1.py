import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join, abspath, dirname
from sklearn import neighbors
from sklearn.svm import SVR
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
from matplotlib import pyplot

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

        date_col = []
        year_start = date(int(file_path[0]),1, 1)

        for i in range(len(file_data)):
            actual_date = year_start + timedelta(days=i)
            date_col.append([actual_date])

        file_data = np.hstack((file_data, date_col))

        if(len(data) == 0):
            data = file_data
        else:            
            data = np.concatenate((data, file_data))
            
    return data

def knn_regression(data, n_neighbors=6, train_test_percentage=20):
    train_test_size = int(len(data) * train_test_percentage / 100)
    train_data, test_data = data[0:train_test_size, :], data[train_test_size:len(data) - 1, :]

    train_X = train_data[:, 1:6]
    train_y = train_data[:,0]
    test_X = test_data[:, 1:6]
    test_y = test_data[:,0]

    knn = neighbors.KNeighborsRegressor(n_neighbors, p=2, weights='uniform')
    print knn.fit(train_X, train_y).score(test_X, test_y)

def svm_regression(data, train_test_percentage=20):
    train_test_size = int(len(data) * train_test_percentage / 100)
    train_data, test_data = data[0:train_test_size, :], data[train_test_size:len(data) - 1, :]

    train_X = train_data[:, 1:6]
    train_y = train_data[:,0]
    test_X = test_data[:, 1:6]
    test_y = test_data[:,0]

    clf = SVR(C=1.0, epsilon=0.2, gamma=0.01)
    print clf.fit(train_X, train_y).score(test_X, test_y)

def autoregression(data, train_test_percentage=20):
    train_test_size = int(len(data) * train_test_percentage / 100)
    train, test = data[0:train_test_size], data[train_test_size:]

    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    print('Test MSE111: %.3f' % model.score(test))
    window = model_fit.k_ar
    coef = model_fit.params

    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()

    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window, length)]
        yhat = coef[0]

        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        
    mse_error = mean_squared_error(test, predictions)
    return predictions

def simple_moving_average(data, n):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=n:
            moving_ave = (cumsum[i] - cumsum[i-n])/n
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

train_test_percentage = 20
weather_data = load_weather_data()
time_series = weather_data[:, 0]
running_mean = running_mean(time_series, 5)[0:20,]

# knn_regression(weather_data, train_test_percentage)
# svm_regression(weather_data, train_test_percentage)
autoregression(time_series, train_test_percentage)