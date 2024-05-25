from utils import *
from dper import *
from dperc import *

def experiment(X,y,Z,missing_rate,run_times):
  p = X.shape[1]
  G = len(np.unique(y))
  q = Z.shape[1]
  S0 = np.array([np.cov(X[y==g], rowvar = False) for g in range(G)])
  Ss = []
  er = []
  for i in range(run_times):
    Xnan = generate_nan(X,missing_rate)

    #DPER
    Sdper = np.array([DPER(Xnan[y==g]) for g in range(G)])

    #DPERC
    Sdperc = np.array([np.zeros((p,p)) for g in range(G)])
    for g in range(G):
      Xg = Xnan[y==g]
      yg = y[y==g]
      Zg = Z[y==g]
      Sdperc[g] = diagonal_matrix(Sdper[g]) + DPERC(Xg,yg,Zg)

    #KNNI
    Xknn = KNN(k=3).fit_transform(Xnan)
    Sknn = np.array([np.cov(Xknn[y==g], rowvar = False) for g in range(G)])

    #MissForest
    Xm = pd.DataFrame.from_records(np.concatenate((Xnan,Z),axis = 1))
    Missf = MissForest()
    XMissf_df = Missf.fit_transform(Xm.iloc[:, :p])
    XMissf = XMissf_df.to_numpy()
    SMissf = np.array([np.cov(XMissf[y==g], rowvar = False) for g in range(G)])

    #Soft-Impute
    XSoft = SoftImpute(max_iters = 20, verbose = False).fit_transform(Xnan)
    SSoft = np.array([np.cov(XSoft[y==g], rowvar = False) for g in range(G)])

    #MICE
    XMice = IterativeImputer(max_iter=20).fit(Xnan).transform(Xnan)
    SMice = np.array([np.cov(XMice[y==g], rowvar = False) for g in range(G)])

    er.append([error(S0,Sdperc), error(S0,Sdper), error(S0,SMissf), error(S0,Sknn), error(S0,SSoft), error(S0,SMice)])
    Ss.append([S0, Sdperc, Sdper ,SMissf, Sknn, SSoft, SMice])
  return (Ss,er)



