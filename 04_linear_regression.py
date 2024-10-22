import numpy as np
from sklearn.linear_model import LinearRegression

#Y=0.5*X-5
X=np.array([0,1,2,3,4,5]).reshape(-1,1)
Y=np.array([-5,-4.5,-4,-3.5,-3,-2.5])

model = LinearRegression()
model.fit(X,Y)

print(model.predict([[8]]))


print("slope:",model.coef_[0],"intercept:",model.intercept_)
