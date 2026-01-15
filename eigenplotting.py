import numpy as np
import math
import skink as skplt
from tqdm import tqdm
import os
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as plt


# correlated 2D gaussian
nsample = 1_000
std = 1
mean = [-1,1]
cov = [[std,0.5*std],
       [0.5*std,std]]


Gauss = np.random.multivariate_normal(mean = mean,
                                      cov = cov,
                                      size = nsample)

xmean = np.mean(Gauss[:,0])
ymean = np.mean(Gauss[:,1])

means = np.array([xmean, ymean])

# scatter plot
fig = plt.figure(1, figsize = (4,4))
plt.scatter(Gauss[:,0],Gauss[:,1], s = 1)
plt.draw()
plt.savefig("testingsss.png")
plt.savefig("testingsss.pdf")
plt.clf()


print("-------------------------------------------")
print("X:")
print(" -> mean = "+str(np.mean(Gauss[:,0])))
print(" -> std =  "+str(np.std(Gauss[:,0])))
print("Y:")
print(" -> mean = "+str(np.mean(Gauss[:,1])))
print(" -> std =  "+str(np.std(Gauss[:,1])))
print("-------------------------------------------")

# Eignevalue decomposition

e, V = np.linalg.eigh(cov)

D = np.diag(1/np.sqrt(e))

print(e)
print(V)

U = D @ V.T
U2 = D @ V

print(np.abs(U @ cov @ U.T - np.eye(2)))
print(np.abs(U2 @ cov @ U2.T - np.eye(2)))
raise("HI")
#transform (x,y)->(m,n)
Gauss_T = []

for i in range(Gauss[:,0].shape[0]):
    Gauss_T.append(D @ V @ (Gauss[i] - means))

Gauss_T = np.array(Gauss_T)

fig = plt.figure(1, figsize = (4,4))
plt.scatter(Gauss_T[:,0],Gauss_T[:,1], s = 1)
plt.draw()
plt.savefig("testingsss2.png")
plt.savefig("testingsss2.pdf")
plt.clf()

print("X:")
print(" -> mean = "+str(np.mean(Gauss_T[:,0])))
print(" -> std =  "+str(np.std(Gauss_T[:,0])))
print("Y:")
print(" -> mean = "+str(np.mean(Gauss_T[:,1])))
print(" -> std =  "+str(np.std(Gauss_T[:,1])))
print("-------------------------------------------")
print(np.cov(Gauss.T))
print(np.cov(Gauss_T.T))
