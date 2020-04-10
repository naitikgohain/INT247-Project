'''import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers'''
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
from keras import Sequential
from keras.layers import Dense

import pickle

#splitting data between train and test
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__=="__main__":
  #reading the data from the csv file
  dataset = pd.read_csv('data1.csv')  

  x= dataset.iloc[:,0:5]
  y= dataset.iloc[:,5]

  #Train test splitting
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

  #Starting our model
  classifier = Sequential()
  #First Hidden Layer
  classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=5))
  #Second  Hidden Layer
  classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
  #Output Layer
  classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
  #Compiling the neural network
  classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
  classifier.fit(X_train,y_train, batch_size=10, epochs=100)

  #Dumping model in a file
  file1 = open('model_neural.pkl', 'wb')
  pickle.dump(classifier, file1)



  '''df=pd.read_csv('data.csv')


  #getting splitted data from function
  train,test = data_split(df, 0.3)

  # splitting features and output for prediction
  x_train = train[['fever','bodypain','age','runnynose','diffbreath']].to_numpy()
  x_test = test[['fever','bodypain','age','runnynose','diffbreath']].to_numpy()

  y_train = train[['infected']].to_numpy().reshape(train.shape[0],)
  y_test = test[['infected']].to_numpy().reshape(test.shape[0],)

  clf=LogisticRegression();
  clf.fit(x_train, y_train);

  file1 = open('model.pkl', 'wb')
  pickle.dump(clf, file1)
'''
 



  '''train, test = train_test_split(df, test_size=0.2)
  train, val = train_test_split(train, test_size=0.2)
  print(len(train), 'train examples')
  print(len(val), 'validation examples')
  print(len(test), 'test examples')

  batch_size = 5 # A small batch sized is used for demonstration purposes
  train_ds = df_to_dataset(train, batch_size=batch_size)
  val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
  test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

  for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )'''
'''
  #example_batch = next(iter(train_ds))[0]

  #def demo(feature_column):
  #  feature_layer = layers.DenseFeatures(feature_column)
  #  print(feature_layer(example_batch).numpy())

  age = feature_column.numeric_column("age")
  diffb = feature_column.numeric_column("diffbreath")
  #demo(age)

  feature_columns = []

  for header in ['fever', 'bodypain', 'runnynose']:
    feature_columns.append(feature_column.numeric_column(header))

  age_buckets = feature_column.bucketized_column(age, boundaries=[12, 20, 30, 40, 50, 60, 70, 82])
  feature_columns.append(age_buckets)
  
  diffb_buckets = feature_column.bucketized_column(diffb, boundaries=[-1,0,1])
  feature_columns.append(diffb_buckets)


  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

  batch_size = 1999
  train_ds = df_to_dataset(train, shuffle=True)
  val_ds = df_to_dataset(val, shuffle=True)
  test_ds = df_to_dataset(test, shuffle=True)

  model = tf.keras.Sequential();
  model.add(feature_layer);
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dense(1))

  model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  metrics=['accuracy'])

  model.fit(train_ds,validation_data=val_ds, epochs=100)

  #model.save('model');
  print(test_ds);
  pred = model.predict(test_ds)
  #print(pred)
  #file1 = open('model.pkl', 'wb')

  # dump information to that file
  #pickle.dump(model, file1)'''


