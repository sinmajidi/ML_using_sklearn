
from sklearn.datasets import load_breast_cancer
bc=load_breast_cancer()

print(bc.data[1])
print(bc.target[1])

# # split train/test
x=bc.data
y=bc.target

from sklearn.model_selection import train_test_split
spliter=train_test_split
x_train,x_test,y_train,y_test=spliter(x,y,test_size=0.2)


# # accuracy
def accuracy(y_train,y_test,y_predict_train,y_predict_test):
    from sklearn.metrics import accuracy_score, confusion_matrix \
        , precision_score, recall_score
    acc_score_train=accuracy_score(y_true=y_train,y_pred=y_predict_train)
    acc_score_test=accuracy_score(y_true=y_test,y_pred=y_predict_test)
    precision_score=precision_score(y_test,y_predict_test)
    recall_score=recall_score(y_test,y_predict_test)
    return acc_score_train,acc_score_test,precision_score,recall_score

# normalize
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# make the model naive bayes
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
y_predict_train=model.predict(x_train)
y_predict_test=model.predict(x_test)
print(f'naive bayes: {accuracy(y_train,y_test,y_predict_train,y_predict_test)}')

# make the model KNN
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_predict_train=model.predict(x_train)
y_predict_test=model.predict(x_test)
print(f'KNN: {accuracy(y_train,y_test,y_predict_train,y_predict_test)}')

# make the model Decision Tree
from sklearn.tree import DecisionTreeClassifier
model1=DecisionTreeClassifier(max_depth=30,min_samples_split=5,min_samples_leaf=1)
model1.fit(x_train,y_train)
y_predict_train=model1.predict(x_train)
y_predict_test=model1.predict(x_test)
print(f'Decision Tree: {accuracy(y_train,y_test,y_predict_train,y_predict_test)}')

# make the model Random Forest
from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(n_estimators=64,max_depth=30,min_samples_split=5,min_samples_leaf=2)
model2.fit(x_train,y_train)
y_predict_train=model2.predict(x_train)
y_predict_test=model2.predict(x_test)
print(f'Random Forest: {accuracy(y_train,y_test,y_predict_train,y_predict_test)}')


# make the model SVC
from sklearn.svm import SVC
model=SVC(kernel='poly')
model.fit(x_train,y_train)
y_predict_train=model.predict(x_train)
y_predict_test=model.predict(x_test)
print(f'SVC: {accuracy(y_train,y_test,y_predict_train,y_predict_test)}')

# make the model ANN
from sklearn.neural_network import MLPClassifier
model=MLPClassifier(hidden_layer_sizes=512,batch_size=64)
model.fit(x_train,y_train)
y_predict_train=model.predict(x_train)
y_predict_test=model.predict(x_test)
print(f'ANN: {accuracy(y_train,y_test,y_predict_train,y_predict_test)}')