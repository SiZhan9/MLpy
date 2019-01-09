# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:00:19 2019

@author: zhang
numpy tutorial
"""
import numpy as np
import matplotlib.pyplot as plt
mu,sigma = 2,0.5
v =np.random.normal(mu,sigma,10000)
plt.hist(v, bins = 100,density =1)
plt.show()