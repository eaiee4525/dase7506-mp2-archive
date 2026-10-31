import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from . import decision as D
L2=.01
def group_logits(theta,x,g,pooling):
 candidate=x@theta[:-1];mask=g.sum(2)>0;size=(g>0).sum(2);freq=np.log(np.maximum(size,1))
 if pooling=='mean':s=np.einsum('ngc,nc->ng',g,candidate);responsibility=g
 else:
  raw=np.where(g>0,candidate[:,None,:],-np.inf);z=logsumexp(raw,axis=2);z=np.where(mask,z,0);s=z-np.log(np.maximum(size,1));responsibility=np.where(g>0,np.exp(np.where(g>0,raw-z[:,:,None],-np.inf)),0)
 return s+theta[-1]*freq,responsibility,freq,mask

def objective(theta,x,g,target,pooling):
 s,r,freq,mask=group_logits(theta,x,g,pooling);logp=np.where(mask,s,-np.inf);logp-=logsumexp(logp,axis=1,keepdims=True);p=np.where(mask,np.exp(logp),0);ds=(p-target)/len(x)
 loss=-(target*np.where(mask,logp,0)).sum()/len(x)+L2/2*float(theta[:-1]@theta[:-1])+.01/2*(theta[-1]-1)**2
 dc=np.einsum('ng,ngc->nc',ds,r);grad=np.r_[np.einsum('nc,ncd->d',dc,x)+L2*theta[:-1],(ds*freq).sum()+.01*(theta[-1]-1)]
 return float(loss),grad

def fit(x,g,y,pooling,start):
 target,mask=D.targets(y,g);res=minimize(objective,start,args=(x,g,target,pooling),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-10,'gtol':1e-6})
 if not np.isfinite(res.fun):raise ValueError('nonfinite')
 return res.x,{'success':bool(res.success),'message':str(res.message),'iterations':int(res.nit),'objective':float(res.fun),'max_gradient':float(abs(res.jac).max()),'frequency_coefficient':float(res.x[-1])}
