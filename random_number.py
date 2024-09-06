import numpy as np
np.random.seed(123) # set the seed for the random number generator
x = np.random.random() # uniform (0,1)
y = np.random.randint(5,9) # discrete uniform 5,...,8
z = np.random.randn(4) # array of four standard normals
print(x,y,'\n',z)


