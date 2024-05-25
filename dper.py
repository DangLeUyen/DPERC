from utils import *

#For single class
def sig_estimate(X,mus0,mus1):
  m=n=l=sig11=sig22=s11=s12=s22=0
  del_case=0
  for i in X.T:
    if np.isfinite(i[0]) and np.isfinite(i[1]):
      m=m+1
      s11=s11+(i[0]-mus0)**2
      s22=s22+(i[1]-mus1)**2
      s12=s12+(i[0]-mus0)*(i[1]-mus1)
      sig11=sig11+(i[0]-mus0)**2
      sig22=sig22+(i[1]-mus1)**2
      del_case=del_case+(i[0]-mus0)*(i[1]-mus1)
    elif np.isfinite(i[0]) and np.isnan(i[1]):
      n=n+1
      sig11=sig11+(i[0]-mus0)**2
    elif np.isnan(i[0]) and np.isfinite(i[1]):
      l=l+1
      sig22=sig22+(i[1]-mus1)**2
  del_case = max(del_case/(m-1),0)
  sig11=sig11/(m+n)
  sig22=sig22/(m+l)
  sig12=solving(-m,s12,(m*sig11*sig22-s22*sig11-s11*sig22),s12*sig11*sig22,del_case)
  return sig11,sig22,sig12

def DPER(X):
  sig=np.zeros((X.shape[1],X.shape[1]))     #estimated covariance matrix
  #estimation of mean
  mu=np.nanmean(X,axis=0)
  #estimation of covariane
  for a in range(X.shape[1]):
    for b in range(a):
      temp=sig_estimate(np.array([X[:,b],X[:,a]]),mu[b],mu[a])
      sig[b][b]=temp[0]
      sig[a][a]=temp[1]
      sig[b][a]=sig[a][b]=temp[2]
  return sig


#for multi-class
def sigma_m(X,y):
  del_case=0
  res=np.array([0]*8)  # [m,n,l,s11,s12,s22,sig11,sig22]
  G=len(np.unique(y))
  mus = [np.nanmean(X[y==g],axis = 0) for g in range(G)]
  for g in range(G):
    m=n=l=s11=s12=s22=sig11=sig22=0
    mus0=mus[g][0]
    mus1=mus[g][1]
    Xg=(X)[y==g]
    for i in Xg:
      if np.isfinite(i[0]) and np.isfinite(i[1]):
        m=m+1
        s11=s11+(i[0]-mus0)**2
        s22=s22+(i[1]-mus1)**2
        s12=s12+(i[0]-mus0)*(i[1]-mus1)
        sig11=sig11+(i[0]-mus0)**2
        sig22=sig22+(i[1]-mus1)**2
      elif np.isfinite(i[0]) and np.isnan(i[1]):
        n=n+1
        sig11=sig11+(i[0]-mus0)**2
      elif np.isnan(i[0]) and np.isfinite(i[1]):
        l=l+1
        sig22=sig22+(i[1]-mus1)**2
    res = res+np.array([m,n,l,s11,s12,s22,sig11,sig22])
  m,n,l,s11,s12,s22,sig11,sig22 = res
  del_case = max(0,del_case/(m-1))
  sig11=sig11/(m+n)
  sig22=sig22/(m+l)
  sig12=solving(-m,s12,(m*sig11*sig22-s22*sig11-s11*sig22),s12*sig11*sig22,del_case)
  return sig11,sig22,sig12

def DPERm(X,y):            #with assumption of equal covariance matrices
  numlabel=len(np.unique(y))        #number of unique label in y
  p=X.shape[1]
  sig=np.zeros((p,p))               #estimated covariance matrix
  #compute mu_est
  mu=np.array([np.nanmean(X[y==g],axis = 0) for g in range(numlabel)])
  #estimation of covariane matrix
  for a in range(p):
    for b in range(a):
      temp=sigma_m(np.array([X[:,b],X[:,a]]),y)
      sig[b][b]=temp[0]
      sig[a][a]=temp[1]
      sig[b][a]=sig[a][b]=temp[2]
  return sig