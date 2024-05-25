from utils import *
from dper import *

def sigma_c(fi,fj,y,Z):
  '''fi,fj: continuous features with NaN
   y: class
   Z: categorical features
  '''
  X = np.array([fi,fj]).T
  G = len(np.unique(y))
  q = Z.shape[1]   #number of categorical features
  sigma = 0
  mus = np.nanmean(X,axis = 0)
  SCD = DPER(X)
  temp = np.array([])
  for i in range(q):
    c = Z[:,i]                #Consider categorical feature i
    Gc = len(np.unique(c))
    mean_class = np.array([np.nanmean(X[c==l],axis = 0) for l in range(Gc)])
    distance_class = 0
    for l in range(Gc):
      mean_class[l] = np.where(~np.isnan(mean_class[l]), mean_class[l],0)  #If mean_class == NaN then mean_class = 0
      distance_class += (sum(c==l))*np.dot(np.dot((mean_class[l]-mus).T,inv(SCD)),(mean_class[l]- mus))
    temp = np.append(temp,distance_class)
  key = np.argmin(temp)
  sigma = sigma_m(X,Z[:,key])[2]
  return sigma

def DPERC(X,y,Z):
  p = X.shape[1]
  res = np.zeros((p,p))
  for i in range(p):
    for j in range(i+1,p,1):
      res[i,j] = res[j,i] =  sigma_c(X[:,i],X[:,j],y,Z)
  return res