import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.neural_network import MLPClassifier



def check_model(y_train,y_test,y_pred_train,y_pred_test):
    result={
        'accuracy_train':accuracy_score(y_train,y_pred_train),
        'accuracy_test':accuracy_score(y_test,y_pred_test)
    }
    return result
    
# 1. Load the dataset
df = pd.read_csv('/datasets/predictive_maintenance_data.csv')
print(df.head())
print(df.shape)
print(df['FailureType'].value_counts())


x=df.drop(['FailureType'],axis=1)
y=df['FailureType']
x=np.array(x)
y=np.array(y)
print(x.shape)
print(y.shape)

scaler=MinMaxScaler(feature_range=(0,1))
x=scaler.fit_transform(x)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)

model=MLPClassifier(hidden_layer_sizes=(128,128),max_iter=500)
model.fit(x_train,y_train)

y_pred_train=model.predict(x_train)
y_pred_test=model.predict(x_test)



for i in range(10):
    print(x_test[i],y_test[i],y_pred_test[i])
    print('\n')

print(check_model(y_train,y_test,y_pred_train,y_pred_test))
