"""Independent unchanged copy of the previously registered 11-feature map."""
import numpy as np

def meta_features(a,b,groups,valid,sd):
    z=[];margins=[]
    for s,scale in zip((a,b),sd):
        center=(s*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True)
        zz=(s-center)/scale;z.append(zz)
        g=np.einsum('ngc,nc->ng',groups,zz)
        g=np.where(groups.sum(2)>0,g,-np.inf);g=np.sort(g,axis=1)
        # Single-group margins have no selection information, set to zero.
        margin=np.where((groups.sum(2)>0).sum(1)>=2,g[:,-1]-g[:,-2],0)
        margins.append(np.clip(margin,0,4)[:,None])
    a,b=z;ma,mb=margins
    freq=np.einsum('ngc,ng->nc',(groups>0).astype(float),(groups>0).sum(2))/4
    ng=(groups.sum(2)>0).sum(1)[:,None]/4
    ng=np.repeat(ng,4,axis=1)
    return np.stack([a,b,a*ma,b*ma,a*mb,b*mb,a*freq,b*freq,freq,a*ng,b*ng],2)

