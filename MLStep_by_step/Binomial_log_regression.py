import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris=datasets.load_iris()
X=iris.data[:,:2]
y=(iris.target !=0)*1

plt.figure(figsize=(6,6))
plt.scatter(X[y==0][:,0],X[y==0][:,1],color='g',label='0')
plt.scatter(X[y==1][:,0],X[y==1][:,1],color='y',label='1')
plt.legend()
plt.show()