# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:23:38 2019

@author: zhang
matplot tutorial for data visiualization
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,5,0.1);
y = np.sin(x)
plt.plot(x,y)
plt.grid(True)