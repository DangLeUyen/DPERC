import time
import math
import pandas as pd
import numpy as np
import requests
from io import StringIO
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from fancyimpute import SoftImpute, KNN
from missforest.missforest import MissForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from numpy.linalg import norm, inv, det



def error(sig, sig_est):
  p = sig.shape[0]
  return norm(sig_est.flatten()-sig.flatten())/(p*p)

def normalize_data(X):
  scaler = StandardScaler()
  scaler.fit(X)
  return scaler.transform(X)

def generate_nan(X, missing_rate):
    X_copy=np.copy(X)
    X_non_missing = X_copy[[0],:]
    X_missing = X_copy[[i for i in range(1,X.shape[0],1)],:]
    XmShape = X_missing.shape
    np.random.seed(2)
    na_id = np.random.randint(0, X_missing.size, round(missing_rate * X_missing.size))
    X_nan = X_missing.flatten()
    X_nan[na_id] = np.nan
    X_nan = X_nan.reshape(XmShape)
    return np.vstack((X_non_missing, X_nan))

def sigCD(X):
    X = X[~np.isnan(X).any(axis=1)]
    mean_vector = np.mean(X, axis=0)
    centered_matrix = X - mean_vector
    covariance_matrix = np.dot(centered_matrix.T, centered_matrix) / (centered_matrix.shape[0] - 1)
    return covariance_matrix

def solving(a,b,c,d,del_case):
  roots = np.roots([a,b,c,d])
  real_roots = np.real(roots[np.isreal(roots)])
  if len(real_roots)==1:
    return real_roots[0]
  else:
    f = lambda x: abs(x-del_case)
    F=[f(x) for x in real_roots]
    return real_roots[np.argmin(F)]
  
def diagonal_matrix(X):
    n = len(X)
    Y = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Y[i][i] = X[i][i]
    return Y

def convert_cov_to_corr(cov_mx):
  corr_mx = cov_mx / np.outer(np.sqrt(np.diag(cov_mx)), np.sqrt(np.diag(cov_mx)))
  return corr_mx


def cal_sub(mx1, mx2):
   diff = np.abs(mx1 - mx2)
   return diff

def cal_mse(mx1, mx2):
   squared_diff = (mx1 - mx2)**2
   return squared_diff