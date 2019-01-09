# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:36:29 2019

@author: zhang
First Chapter: Linear Regression with graident descent and normal equation

"""
import numpy as np
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
    return J


## gradientDescent  Method
def gradientDescent(X,Y,theta,alpha,num_iters):
    m = len(Y)
    n = len(theta)
    J = []
    i = 0
    theta_collect =[]
    ## iteration algorithm
    while i <= num_iters:
        Dev = np.transpose(theta) @ X - Y ## Dev is a 1*m vector
        new_theta = theta - (X @ np.transpose(Dev))*alpha/m
        theta = new_theta
        J.append( computeCost(X,Y,theta))
        theta_collect.append(theta)
        i += 1
    return J, theta, theta_collect

## Main Function
X = np.array([[1,1,1,1],[2014,1416,1534,852],[5,3,3,2],[1,2,2,1],[45,40,30,36]])
Y = np.array([460,232,315,178])
m = len(Y)
n = len(X)
theta = np.ones(n)
alpha = 10
num_iters = 20
J_collect= []
theta_list = []
J_collect,final_theta,theta_list = gradientDescent(X,Y,theta,alpha,num_iters) 