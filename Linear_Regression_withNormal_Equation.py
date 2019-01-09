# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 18:12:29 2019

@author: zhang
first Chapter for Linear Regression with Normal equation
"""
import numpy as np
#import matplotlib.pyplot as plt
def computeCost(X,Y,theta):
    m = len(Y)
    J = 0
    dev = np.transpose(theta) @ X - Y
    J = dev @ np.transpose(dev)/(2*m)
    print(J)
    return J

def Normalequation(X,Y):
    X_prime = np.transpose(X)
    temp = X_prime @ X
    temp_inv = np.linalg.inv(temp)
    theta = temp_inv @ X_prime @ Y
    print(theta)
    return theta

X = np.array([[1,1,1,1],[2014/2014,1416/2014,1534/2014,852/2014],[5,3,3,2],[1,2,2,1],[45,40,30,36]])
X_prime= np.transpose(X)
##X = np.array([[1,1,1,1],[3,2,5,4],[5,3,6,7],[12,17,21,23]])
##Y = np.array([3,5,7,9])
Y = np.array([460,232,315,178])
Y_prime = np.transpose(Y)
m = len(Y)
n = len(X)
theta = np.zeros(n).reshape(n,1)
#alpha = 0.00001
#num_iters = 300
#J_collect= []
#J_collect,final_theta,counter_iters = gradientDescent(X,Y,theta,alpha,num_iters) 
Cost_Func_init = computeCost(X,Y,theta)
print(Cost_Func_init)
theta_final = Normalequation(X_prime,Y_prime)
Cost_Func_final = computeCost(X,Y,theta_final)
print(Cost_Func_final)
#plt.figure('model')
#for each in J_collect:
#    Cost_Data.append(each[0,0])

#plt.plot(counter_iters,Cost_Data)
#plt.xlabel('Times of iteration')
#plt.ylabel('Cost Function Values')
#plt.show()
