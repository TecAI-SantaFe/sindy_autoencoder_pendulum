import numpy as np
import os


cwd = os.getcwd()

X = np.load(cwd+'/X.npy')
Xdot = np.load(cwd+'/Xdot.npy')
Xddot = np.load(cwd+'/Xddot.npy')
Y = np.load(cwd+'/Y.npy')
print(Y.shape)

da = 5.e-2

min1 = -np.pi/2-da
max1 = -np.pi/2+da

min2 = -da
max2 = da

min3 = np.pi/2-da
max3 = np.pi/2+da

mask = np.logical_or(np.logical_or(np.logical_and(Y>min1,Y<max1),
                                   np.logical_and(Y>min2,Y<max2))
                     ,np.logical_and(Y>min3,Y<max3))
Yfilt = np.extract(mask.astype(bool), Y) 
print(Yfilt.shape)

#count = 0
#for n in range(Y.shape[0]):
#    for i in range(Y.shape[1]):
#        ang = Y[n,i]
#        if (ang>min1 and ang<max1) or (ang>min2 and ang<max2) or (ang>min3 and ang<max3):
#            count += 1
#print(count)