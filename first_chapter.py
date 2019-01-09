# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:36:29 2019

@author: zhang
First Chapter: Linear Regression with graident descent and normal equation

"""
import numpy as np
import matplotlib.pyplot as plt
###
## X is a n*m matrix, m is trainning samples. n is attribute for very samples, the first attribute is 1 for theta zero
## Y is a 1*m  vector as m trainning resuslt
## theta is a n*1 vector
## Calculate the cost Function
def computeCost(X,Y,theta):
    m = len(Y)
    J = 0
    dev = np.transpose(theta) @ X - Y
    J = dev @ np.transpose(dev)/(2*m)
    print(J)
    return J


## gradientDescent  Method
def gradientDescent(X,Y,theta,alpha,num_iters):
    m = len(Y)
#    n = len(theta)
    J = []
    i = 0
    counter_iters =[]
    ## iteration algorithm
    while i <= num_iters:
        Dev = np.transpose(theta) @ X - Y ## Dev is a 1*m vector
        print(Dev)
        J.append( computeCost(X,Y,theta))
        new_theta = theta - (X @ np.transpose(Dev))*alpha/m
        theta = new_theta
        print(new_theta)
        i += 1
        counter_iters.append(i)
        
    return J, theta, counter_iters

## Main Function

X = np.array([[1,1,1,1],[2014/2014,1416/2014,1534/2014,852/2014],[5,3,3,2],[1,2,2,1],[45,40,30,36]])
##X = np.array([[1,1,1,1],[3,2,5,4],[5,3,6,7],[12,17,21,23]])
##Y = np.array([3,5,7,9])
Y = np.array([460,232,315,178])
m = len(Y)
n = len(X)
theta = np.zeros(n).reshape(n,1)
alpha = 0.00001
num_iters = 300
J_collect= []
J_collect,final_theta,counter_iters = gradientDescent(X,Y,theta,alpha,num_iters) 
Cost_Data = []
plt.figure('model')
for each in J_collect:
    Cost_Data.append(each[0,0])

plt.plot(counter_iters,Cost_Data)
plt.show()