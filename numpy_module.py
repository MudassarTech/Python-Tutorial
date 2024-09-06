
from re import T
from numpy import arange, dtype, array, cos, dtype, integer, ndarray , square , sqrt, cosh
import numpy as np
from timeit import Timer, timeit
import numpy.linalg

# x = array ([1.2 ,2.3 ,4.5])
# def stat(x):
#    n = len(x) #the length of x
#    meanx = sum(x)/n
#    stdx = sqrt(sum(square(x - meanx))/n)
#    return [meanx ,stdx]
# print(stat(x))
# numpy.info(x)
# array = [2,3,4,5,6,7]
# array.insert(2,4)
# array.pop()
# print(array)
# print(sqrt(4))
# print(cosh(45))
# print(cos(345))

ans = 'y'
while ans != 'n':
   outcome = np.random.randint (1 ,6+1)
   if outcome == 6:
       print("Hooray a 6!")
       break
   else:
       print("Bad luck , a", outcome)
       ans = input("Again? (y/n) ")


# A = {3, 2, 2, 4}
# B = {4, 3, 1}
# C = A & B
# for i in A:
#     print(i)
# print(C)

# setA = {3, 2, 4, 2,5,5}
# setB = {x**2 for x in setA}    #duplicate number not multiple
# print(setB)
# listA = [3, 2, 4, 2]
# listB = [x**2 for x in listA]
# print(listB)


 
# x = np.cos(90)
# data = [1,2,3,4,5,6,3,6]
# y = np.mean(data)
# z = np.std(data)
# # print('cos(1) = {0:1.8f} mean = {1} std = {2}'.format(x,y,z))
# print(f"cos(90) = {x} mean = {y} std = {z}")
# print(np.average(data))


# a = np.array([[5, 9, 13], [14, 10, 12], [11, 15, 19]]) 
# m = np.mean(a)  
# print(m)    #   12.0 
# z = np.mean(a, where=[[True], [False], [False]])    # 9.0
# print(z)
# print(np.std(a, dtype=np.float16))





# a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
# m = mean = np.mean(a, axis=1, keepdims=True)
# print(m)


g = globals()
n = 100000
t1 = timeit( "std = np.std(a, axis=1, mean= mean)",globals=g, number=n)
t2 = timeit("std = np.std(a, axis=1)", globals=g, number=n)
print(f'Percentage execution time saved {100*(t2-t1)/t2:.2f}%')


# A = np.zeros([2,3,2]) # 2 by 3 by 2 array of zeros
# print(A)
# print(A.shape)      #Number of rows and coloumn
# print(type(A))         # A is an ndArray



# a = np.array(range(0,50,2)) # equivalent to np.arange(4)
# b = np.array([0,1,2,3])
# C = np.array([[1,2,3],[3,2,1]])
# print(a, '\n', b,'\n', C)
# print(np.arange(20))


# a = np.array(range(16)) #a is an ndarray of shape (9,)
# print(a.shape)
# A = a.reshape(8,2) #A is an ndarray of shape (3,3)
# print(a)
# print(A)


# Transpose
# a = np.arange(3)    #1D array (vector) of shape (3,)
# print(a)
# print(a.shape)
# b = a.reshape(3,1)     # 3x1 array (matrix) of shape (3,1)
# print(b)
# print(b.T)
# A = np.arange(9).reshape(3,3)
# print(A.T)



# A= np.ones((3,3))
# B = np.zeros((3,3))
# H = np.hstack((A,B))
# V = np.vstack((A,B))
# C = np.concatenate((A,B))
# S = np.stack((A,B))
# print(C,'\n',V,'\n',H,'\n',S)

# for extra
# a = np.ma.arange(3)
# a[1] = np.ma.masked
# b = np.arange(2, 5)
# print(a)



# masked_array(data=[0, --, 2],
#              mask=[False,  True, False],
#              fill_value=999999)
# print(b)
# array([2, 3, 4])
# np.concatenate([a, b])
# a = masked_array(data=[0, 1, 2, 2, 3, 4],
#              mask=False,
#        fill_value=999999)
# np.ma.concatenate([a, b])
# masked_array(data=[0, --, 2, 2, 3, 4],
#              mask=[False,  True, False, False, False, False],
#        fill_value=999999)



# arrays = [np.random.randn(5, 2) for _ in range(10)]
# a = np.stack(arrays, axis=2).shape
# print(a)


# A = np.array(range(9)).reshape(3,3)
# print(A)
# print(A[0]) # first row
# print(A[:,1]) # second column
# print(A[0,1]) # element in first row and second column
# print(A[0:1,1:2]) # (1,1) ndarray containing A[0,1] = 1
# print(A[1:,-1]) # elements in 2nd and 3rd rows, and last column
# A[1:,1] = [2,3] # change two elements in the matrix A above
# print(A)



# x = np.array([[4,2],[3,4]])
# y = np.array([[2,1],[6,2]])
# print(x+y)
# print(np.divide(x,y)) # same as x/y
# print(np.sqrt(y))
# print(np.dot(x,y))
# print(x.dot(x)) # same as np.dot(x,x)
# print(x @ y)



# a = [[2, 0], [5, 1]]
# b = [[3, 4], [4, 2]]
# print(np.arange(3*4*5*6)[::1].reshape((5,4,6,3)))
# print(np.dot(a, b)[1,1])
# a = np.arange(3*4*5*6).reshape((6,4,5,3))
# print(a)
# print(np.dot([2j, 3j], [2j, 3j]))




# A= np.arange(4).reshape(2,2) # (2,2) array
# x1 = np.array([40,500]) # (2,) array
# x2 = x1.reshape(2,1) # (2,1) array                                     
# print(A + x1) # shapes (2,2) and (2,)
# print(A * x2) # shapes (2,2) and (2,1)
# B = np.arange(8).reshape(2,2,2)
# print(B)
# b = np.arange(4).reshape(2,2)
# print(B@b)


# [[ 40 501]
# [ 42 503]]
# [[ 0 40]
# [1000 1500]]
# [[[ 2 3]
# [ 6 11]]
# [[10 19]
# [14 27]]]


# a = np.array(range(9)).reshape(3,3)
# print(a.sum(axis=1,keepdims=True)) #summing over rows gives column totals
# print(a)











