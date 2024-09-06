from pickle import DICT
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot

# series = pd.Series(<data>, index=['index'])

# DICT = {'one':1, 'two':2, 'three':3, 'four':4}
# print(pd.Series(DICT))

# years = ['2000','2001','2002']
# cost = [2.34, 2.89, 3.01]
# print(pd.Series(cost,index = years, name = 'MySeries')) #name it

# pd.DataFrame(<data>, index=['<row_name>'], columns=['<column_name>'])
# ICT = {'numbers':[1,2,3,4], 'squared':[1,4,9,16] }
# df = pd.DataFrame(DICT, index = list('abcd'))
# print(df)


# Manupalting data Frame

# ages = [6,3,5,6,5,8,0,3]
# d={'Gender':['M', 'F']*4, 'Age': ages}
# df1 = pd.DataFrame(d)
# df1.at[0,'Age']= 60               # change an element
# df1.at[1,'Gender'] = 'Female'            # change another element
# df2 = df1.drop('Age',1)                # drop a column
# df3 = df2.copy();                       # create a separate copy of df2
# df3['Age'] = ages                      # add the original column
# dfcomb = pd.concat([df1,df2,df3],axis=1)    # combine the three dfs
# print(dfcomb)

# Useful pandas methods for data manipulation.
# agg:  Aggregate the data using one or more functions.
# apply:  Apply a function to a column or row.
# astype:  Change the data type of a variable.
# concat:  Concatenate data objects.
# replace:  Find and replace values.
# read_csv:   Read a CSV file into a DataFrame.
# sort_values:  Sort by values along rows or columns.
# stack Stack:  a DataFrame.
# to_excel: Write a DataFrame to an Excel file


# d={'Gender':['M', 'F', 'F']*4, 'Age': [6,3,5,6,5,8,0,3,6,6,7,7]}
# df=pd.DataFrame(d)
# print(df.dtypes)
# df['Gender'] = df['Gender'].astype('category') #change the type
# print(df.dtypes)


  #  Extracting Information

# ages = [6,3,5,6,5,8,0,3]
# np.random.seed(123)
# df = pd.DataFrame(np.random.randn(3,4), index = list('abc'),
# columns = list('ABCD'))
# print(df)
# df1 = df.loc["b":"c","B":"C"] # create a partial data frame
# print(df1)
# meanA = df['A'].mean() # mean of 'A'column
# print('mean of column A = {}'.format(meanA))
# expA = df['A'].apply(np.exp) # exp of all elements in 'A'column
# print(expA)


# Useful pandas methods for data inspection.
# columns: Column names.
# count: Counts number of non-NA cells.
# crosstab: Cross-tabulate two or more categories.
# describe:  Summary statistics.
# dtypes: Data types for each column.
# head:  Display the top rows of a DataFrame.
# groupby:  Group data by column(s).
# info:  Display information about the DataFrame.
# loc:  Access a group or rows or columns.
# mean:  Column/row mean.
# plot:  Plot of columns.
# std:  Column/row standard deviation.
# sum:  Returns column/row sum.
# tail:  Display the bottom rows of a DataFrame.
# value_counts:  Counts of different non-null values.
# var:  Variance.

# df = pd.DataFrame({'W':['a','a','b','a','a','b'],
# 'X':np.random.rand(6),
# 'Y':['c','d','d','d','c','c'], 'Z':np.random.rand(6)})
# print(df)
# print(df.groupby('W').mean())
# print(df.groupby(['W', 'Y']).mean())
# print(df.groupby('W').agg([sum,np.mean]))



#  Plotting

# df = pd.DataFrame({'normal':np.random.randn(100),
# 'Uniform':np.random.uniform(0,1,100)})
# font = {'family': 'serif', 'size': 14} #set font
# matplotlib.rc('font', **font) # change font
# df.plot() # line plot (default)
# df.plot(kind = 'box') # box plot
# matplotlib.pyplot.show() #render plots












