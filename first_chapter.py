# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:36:29 2019

@author: zhang
First Chapter: Linear Regression
"""
import numpy as np

## Calculate the cost Function
def computeCost(X,y,theta):
    m = len(y)
    J = 0
    J = np.transpose(X*theta - y)*(X*theta -y)/(2*m)
    return J
# Note there is an additonal line of x for theta zero

## gradientDescent  Method
def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    n = len(theta)
    temp = np.matrix(np.zeros(n,num_iters))