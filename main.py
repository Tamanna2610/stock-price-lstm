from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

microsoft = pd.read_csv('MicrosoftStock.csv')

if 'index' in microsoft.columns:
    microsoft = microsoft.drop(columns=['index'])
    
microsoft['date'] = pd.to_datetime(microsoft['date'])

plt.figure(figsize=(10,4))
plt.plot(microsoft['date'],microsoft['open'],color="blue",label="Open")
plt.plot(microsoft['date'],microsoft['close'],color="green",label="Close")
plt.title("Microsoft Open-Close Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(microsoft['date'],microsoft['volume'],color="red",label='Volume')
plt.title("Microsoft Stock Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()

sns.heatmap(microsoft.select_dtypes(include=np.number).corr(),annot=True,cbar=False)
plt.title("Feature Correlation")
plt.show()

microsoft = microsoft[(microsoft['date']> datetime(2013,1,1))&(microsoft['date']<datetime(2018,1,1))]

close_prices = microsoft[['close']]
dataset = close_prices.values
training_size=int(np.ceil(len(dataset)*0.95))

scaler= StandardScaler()
scaled_data=scaler.fit_transform(dataset)

train_data = scaled_data[0:training_size,:]
x_train,y_train=[],[]

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model= keras.models.Sequential([
    keras.layers.LSTM(64,return_sequences=True,input_shape=(x_train.shape[1],1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(128),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train,y_train,epochs=20,batch_size=32)

test_data=scaled_data[training_size - 60:,:]
x_test=[]
y_test= dataset[training_size:,:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions = model.predict(x_test)

train = microsoft[:training_size]
test = microsoft[training_size:].copy()
test.loc[:,'Predictions']=predictions

plt.figure(figsize=(12, 6))
plt.plot(train['close'], label='Train', color='blue')
plt.plot(test['close'].reset_index(drop=True), label='Actual', color='green')
plt.plot(test['Predictions'].reset_index(drop=True), label='Predicted', color='red')
plt.title("Microsoft Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price (Scaled)")
plt.legend()
plt.show()