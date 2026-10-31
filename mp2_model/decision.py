"""Two finite decisions on fixed candidate scores; independent local implementation."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,expit
from .legacy_meta import meta_features

def group_arrays(questions):
    g=np.zeros((len(questions),4,4),np.float32);v=np.zeros((len(questions),4),np.float32)
    for qi,q in enumerate(questions):
        answers=[]
        for ci,a in enumerate(q.answers):
            if a is None:continue
            v[qi,ci]=1
            if a not in answers:answers.append(a)
        for gi,a in enumerate(answers):
            members=[ci for ci,ans in enumerate(q.answers) if ans==a];g[qi,gi,members]=1/len(members)
    return g,v

def group_features(a,b,g,v,sd,rms,pooling):
    # Exact old feature map first. All-invalid rows are inference-only fallbacks.
    if pooling not in ('mean','lme'):raise ValueError(pooling)
    valid_rows=v.sum(1)>0;features=np.zeros((len(a),4,11),np.float64)
    if valid_rows.any():features[valid_rows]=meta_features(a[valid_rows],b[valid_rows],g[valid_rows],v[valid_rows],sd)
    x=np.einsum('ngc,ncd->ngd',g,features)
    if pooling=='lme':
        deltas=[];margins=[]
        mask=g.sum(2)>0
        for s,scale in zip((a,b),sd):
            center=(s*v).sum(1,keepdims=True)/np.maximum(v.sum(1,keepdims=True),1)
            z=(s-center)/scale;mean=np.einsum('ngc,nc->ng',g,z)
            sorted_g=np.sort(np.where(mask,mean,-np.inf),axis=1)
            margin=np.zeros(len(s),float);multi=mask.sum(1)>=2
            margin[multi]=np.clip(sorted_g[multi,-1]-sorted_g[multi,-2],0,4);margins.append(margin[:,None])
            delta=np.zeros((len(s),4),float)
            for gi in range(4):
                members=g[:,gi]>0;size=members.sum(1);exists=size>0
                delta[exists,gi]=logsumexp(np.where(members[exists],z[exists].astype(float),-np.inf),axis=1)-np.log(size[exists])-mean[exists,gi]
            deltas.append(delta)
        da,db=deltas;ma,mb=margins;freq=(g>0).sum(2)/4;ng=mask.sum(1)[:,None]/4
        x+=np.stack([da,db,da*ma,db*ma,da*mb,db*mb,da*freq,db*freq,np.zeros_like(da),da*ng,db*ng],2)
    return x/rms

def targets(y,g):
    target=np.einsum('ngc,nc->ng',g,y);mask=g.sum(2)>0
    if not np.allclose(target.sum(1),1):raise ValueError('Training requires exactly one correct canonical answer group; zero-correct rows cannot be silently relabelled.')
    return target,mask

def loss_gradient(scores,target,mask,objective):
    n=len(scores)
    if objective=='ce':
        z=np.where(mask,scores,-np.inf);logp=z-logsumexp(z,axis=1,keepdims=True)
        loss=float(-np.where(mask,target* np.where(mask,logp,0),0).sum()/n)
        return loss,(np.where(mask,np.exp(logp),0)-target)/n
    if objective!='pair':raise ValueError(objective)
    good=(scores*target).sum(1,keepdims=True);wrong=mask&(target<.5)
    factor=2/np.maximum(mask.sum(1,keepdims=True),1)
    diff=scores-good;loss=float((np.logaddexp(0,diff)*wrong*factor).sum()/n)
    ds=expit(diff)*wrong*factor/n
    ds-=target*ds.sum(1,keepdims=True)
    return loss,ds

def objective(w,x,target,mask,kind,l2=.01):
    scores=np.einsum('ngd,d->ng',x,w);loss,ds=loss_gradient(scores,target,mask,kind)
    return loss+l2/2*float(w@w),np.einsum('ngd,ng->d',x,ds)+l2*w

def fit(a,b,y,g,v,sd,rms,pooling,kind):
    keep=(g.sum(2)>0).sum(1)>=2
    x=group_features(a[keep],b[keep],g[keep],v[keep],sd,rms,pooling)
    target,mask=targets(y[keep],g[keep]);w0=np.zeros(11);w0[0]=1
    res=minimize(objective,w0,args=(x,target,mask,kind),jac=True,method='L-BFGS-B',
        options={'maxiter':300,'gtol':1e-7,'ftol':1e-10})
    if not np.isfinite(res.fun) or not np.isfinite(res.x).all():raise FloatingPointError('Nonfinite decision fit')
    return res.x,{'success':bool(res.success),'message':str(res.message),'iterations':int(res.nit),
        'objective':float(res.fun),'gradient_max':float(np.abs(res.jac).max()),'fit_questions':int(keep.sum()),
        'pooling':pooling,'loss':kind,'parameter_count':11,'l2':.01,'weights':res.x.tolist()}

def choose(questions,scores,g):
    if scores.shape!=(len(questions),4) or not np.isfinite(scores).all():raise ValueError('Finite group scores required')
    chosen=[]
    for qi,q in enumerate(questions):
        valid=np.flatnonzero(g[qi].sum(1)>0)
        if not len(valid):chosen.append(0);continue
        winner=max(valid,key=lambda gi:(scores[qi,gi],-int(np.flatnonzero(g[qi,gi]>0)[0])))
        chosen.append(int(np.flatnonzero(g[qi,winner]>0)[0]))
    return chosen
