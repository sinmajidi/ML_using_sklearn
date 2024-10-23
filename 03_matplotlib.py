import matplotlib.pyplot as plt
import numpy as np

x= [35, 33, 31, 29]
y= [45, 55, 61, 32]

# plt.plot(x,y,color='red')
# plt.title("test_title")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# plt.scatter(x,y,color="black",label="test")
# plt.legend()
# plt.show()


# x=["red","blue","green"]
# y=[10,12,9]
# plt.title("hello")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.bar(x,y,color='black')
# plt.show()


# x=["red","blue","green"]
# y=[10,12,9]
# plt.pie(y,labels=x)
# plt.show()





x=np.random.rand(1000,1)
print(x)
plt.hist(x, bins=10, edgecolor='black')
plt.show()


