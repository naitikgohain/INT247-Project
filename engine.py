import pandas as pd
import numpy as np

#splitting data between train and test
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#reading the data from the csv file
df=pd.read_csv('data.csv')

#getting splitted data from function
train,test = data_split(df, 0.3)

# splitting features and output for prediction
x_train = train[['fever','bodypain','age','runnynose','diffbreath']].to_numpy()
x_test = test[['fever','bodypain','age','runnynose','diffbreath']].to_numpy()

y_train = train[['infected']].to_numpy().reshape(train.shape[0],)
y_test = test[['infected']].to_numpy().reshape(test.shape[0],)

