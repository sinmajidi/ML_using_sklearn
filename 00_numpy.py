import numpy as np

a=np.array(['1','2','3','4'])
print(a," ",a.shape)
reshaped_a=a.reshape((4,1))
print(reshaped_a," ",reshaped_a.shape)

#if i dont know a dimention i should use -1
reshaped_a=a.reshape((2,-1))
print(reshaped_a," ",reshaped_a.shape)
#-----------------------------------------------------------

# Creating a vector (1D array)
vector = np.array([1, 2, 3, 4, 5])
print("Vector:\n", vector)

# Creating a matrix (2D array)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix:\n", matrix)

# Creating a tensor (3D array)
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Tensor:\n", tensor)

#---------------------------------------------------------

a=np.array([1,2,3])
b=np.array([4,5,6])
print("a+b=",a+b)
print("a-b=",a-b)
print("a*b=",a*b)
print("a+5=",a+5) # data broadcasting


#---------------------------------------------------------

a=np.array([1,2,3,4,5,6,7,8,9,10])
print("a[2:5]=",a[2:5])
print("a[a>3]=",a[a>3])


#---------------------------------------------------------

a=np.array([1,2,3,4,5,6,7,8,9,10])
print(np.min(a))
print(np.max(a))
print(np.sum(a))
print(np.average(a))
print(np.sort(a))

#---------------------------------------------------------

random_np_array=np.random.randint(0,100,(2,2))
print(random_np_array)