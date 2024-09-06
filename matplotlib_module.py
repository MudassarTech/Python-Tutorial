import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d 




#  simple graph
# plt.title( "new figure",loc="center",pad=10.4)
# x = [1,2,3,4,5,6,7,8]
# y = [5,20,3,8,2,4,12,4]
# plt.plot(x,y)
# plt.show()


    # Creating a Basic Plot
# x = np.arange(0, 10, 0.1)
# u = np.arange(0,10)
# y = np.sqrt(x)
# v = u/3
# plt.figure(figsize = (8,5),edgecolor="red",facecolor="yellow")     # size of plot in inches
# plt.plot(x,y,"r--")       # plot green dashed line
# plt.plot(u,v,'bo')       # plot red dots
# plt.xlabel('X',loc="right")
# plt.ylabel('Y',loc="top",fontdict=None)
# plt.tight_layout(h_pad=20,w_pad=10)
# plt.savefig('sqrtplot.pdf',format="pdf")     # saving as pdf
# plt.show()             # both plots will now be drawn


            #  A histogram and scatterplot

# x = np.random.randn(1000)
# u = np.random.randn(100)
# v = np.random.randn(100)
# plt.figure(facecolor="yellow") 
# plt.subplot(121)     # first subplot
# plt.hist(x,bins=15, facecolor='r')
# plt.xlabel('X Variable')
# plt.ylabel('Counts')
# plt.subplot(122)       # second subplot
# plt.scatter(u,v,c='b', alpha=1)
# plt.show()


        #  Three-dimensional scatter- and surface plots.

# def npdf(x,y):
#     return np.exp(-0.5*(pow(x,2)+pow(y,2)))/np.sqrt(2*np.pi)
# x, y = np.random.randn(100), np.random.randn(100)
# z = npdf(x,y)
# xgrid, ygrid = np.linspace(-3,3,100), np.linspace(-3,3,100)
# Xarray, Yarray = np.meshgrid(xgrid,ygrid)
# Zarray = npdf(Xarray,Yarray)
# fig = plt.figure(figsize=plt.figaspect(0.4))
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(x,y,z, c='g')
# ax1.set_xlabel('$x$')
# ax1.set_ylabel('$y$')
# ax1.set_zlabel('$f(x,y)$')
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(Xarray,Yarray,Zarray,cmap='viridis',edgecolor='none')
# ax2.set_xlabel('$x$')
# ax2.set_ylabel('$y$')
# ax2.set_zlabel('$f(x,y)$')
# plt.show()




