import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score
import pickle

df=read_csv=pd.read_csv('datasets/heart_statlog_cleveland_hungary_final.csv')
# print(df.head())
# print(df.shape)
# print(df['target'].value_counts())


x=df.drop('target',axis=1)
y=df['target']
# print(x.shape,y.shape)


x=np.array(x)
y=np.array(y)
# print(x.shape,y.shape)


scaler=StandardScaler()
x=scaler.fit_transform(x)




X_train,X_test,Y_trian,Y_test=train_test_split(x,y,test_size=0.2)
# print(X_train.shape,X_test.shape)


model=DecisionTreeClassifier(max_depth=20,min_samples_split=8,min_samples_leaf=4)
model.fit(X_train,Y_trian)


Y_train_pred=model.predict(X_train)
Y_test_pred=model.predict(X_test)

print('accuracy:',accuracy_score(Y_train_pred,Y_trian)*100,'%')
print('accuracy:',accuracy_score(Y_test_pred,Y_test)*100,'%')


print(model.predict([X_train[10]]))
print(X_train[10].reshape(1,-1))

















model_RF=RandomForestClassifier(n_estimators=100)
model_RF.fit(X_train,Y_trian)


Y_train_pred=model_RF.predict(X_train)
Y_test_pred=model_RF.predict(X_test)

print('accuracy:',accuracy_score(Y_train_pred,Y_trian)*100,'%')
print('accuracy:',accuracy_score(Y_test_pred,Y_test)*100,'%')

print('recall:',recall_score(Y_train_pred,Y_trian)*100,'%')
print('recall:',recall_score(Y_test_pred,Y_test)*100,'%')

print('precision:',precision_score(Y_train_pred,Y_trian)*100,'%')
print('precision:',precision_score(Y_test_pred,Y_test)*100,'%')


print(model_RF.predict([X_train[10]]))
print(X_train[10].reshape(1,-1))



# Save the model using pickle
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the model using pickle
with open('random_forst_model.pkl', 'wb') as file:
    pickle.dump(model_RF, file)



# with open('decision_tree_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)


# # Use the loaded model to make predictions
# sample_data = X_test[0].reshape(1, -1)
# print('Prediction from loaded model:', loaded_model.predict(sample_data))