import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

X,y=make_blobs(300,2,centers=2,random_state=2,cluster_std=1.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='summer')
model_GBN=GaussianNB()
model_GBN.fit(X,y)
rng=np.random.RandomState(0)
Xnew=[-6,-14]+[14,18]*rng.rand(2000,2)
ynew=model_GBN.predict(Xnew)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
lim=plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='summer', alpha=0.1)
plt.axis(lim);
yprob = model_GBN.predict_proba(Xnew)
print(yprob[-10:].round(3))
