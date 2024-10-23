import numpy as np
from sklearn.linear_model import LinearRegression

#Y=2*X-3
X=np.array([0,1,2,3,4,5]).reshape(-1,1)
Y=np.array([-3,-1,1,3,5,7])

model = LinearRegression()
model.fit(X,Y)

print(model.predict([[8]]))


print("slope:",model.coef_[0],"intercept:",model.intercept_)



import matplotlib.pyplot as plt
plt.scatter(X,Y)
# y=mx+z
m=model.coef_[0]
z=model.intercept_
plt.plot(X,model.predict(X))
plt.show()