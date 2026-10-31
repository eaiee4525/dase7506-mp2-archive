"""NumPy analytic gradients for the nine fixed models. No IO, labels at inference."""
import numpy as np
from scipy.special import expit


def loss_grad(scores,y,valid,groups,objective):
    n=len(y)
    if objective=='bce_valid':
        weights=valid/np.maximum(valid.sum(1,keepdims=True),1)/n
        return float(((np.logaddexp(0,scores)-y*scores)*weights).sum()), (expit(scores)-y)*weights
    if objective!='group_ce':raise ValueError(objective)
    gs=np.einsum('ngc,nc->ng',groups,scores)
    mask=groups.sum(2)>0
    gs=np.where(mask,gs,-np.inf)
    maximum=gs.max(1,keepdims=True)
    ex=np.where(mask,np.exp(gs-maximum),0)
    prob=ex/ex.sum(1,keepdims=True)
    target=np.einsum('ngc,nc->ng',groups,y)
    if not np.allclose(target.sum(1),1):raise ValueError('Missing/ambiguous correct group')
    logp=np.where(mask,gs-maximum-np.log(ex.sum(1,keepdims=True)),0)
    loss=float(-(target*logp).sum()/n)
    ds=np.einsum('ngc,ng->nc',groups,(prob-target)/n)
    return loss,ds


def initialize(config,dtype=np.float32):
    rng=np.random.default_rng(config['seed']);p={}
    def weight(name,a,b):p[name]=rng.uniform(-np.sqrt(6/(a+b)),np.sqrt(6/(a+b)),(a,b)).astype(dtype)
    def bias(name,b):p[name]=np.zeros(b,dtype)
    if config['package']=='P1':
        d=config.get('input_dim',1024)
        if config['capacity']=='linear':weight('w',d,1);bias('b',1)
        else:weight('w1',d,64);bias('b1',64);weight('w2',64,1);bias('b2',1)
    else:
        d=config.get('hash_features',8192);k=config.get('text_projection',32)
        weight('e',d,k);bias('eb',k)
        weight('w1',3*k+20,config.get('head_width',32));bias('b1',config.get('head_width',32))
        weight('w2',config.get('head_width',32),1);bias('b2',1)
    return p


def active_masks(p,c):
    masks={k:np.ones_like(v) for k,v in p.items()}
    if c['package']=='P3':
        k=c.get('text_projection',32);condition=c['condition']
        body=condition!='format_final'
        if not body:masks['e']*=0;masks['eb']*=0;masks['w1'][:k]*=0
        if 'question' not in condition:masks['w1'][k:2*k]*=0
        if 'delta' not in condition:masks['w1'][2*k:3*k]*=0
    return masks


def forward(p,c,x,body=None,question=None):
    if c['package']=='P1':
        if c['capacity']=='linear':return (x@p['w']+p['b']).reshape(-1,4),(x,)
        a=x@p['w1']+p['b1'];h=np.maximum(a,0)
        return (h@p['w2']+p['b2']).reshape(-1,4),(x,a,h)
    n=len(x);k=c.get('text_projection',32);condition=c['condition']
    if condition=='format_final':
        a=np.zeros((n,4,k),p['e'].dtype);h=a;aq=np.zeros((n,k),p['e'].dtype);u=aq
    else:
        a=np.asarray(body@p['e']).reshape(n,4,k)+p['eb'];h=np.maximum(a,0)
        if 'question' in condition:aq=np.asarray(question@p['e'])+p['eb'];u=np.maximum(aq,0)
        else:aq=np.zeros((n,k),p['e'].dtype);u=aq
    interaction=h*u[:,None,:] if 'question' in condition else np.zeros_like(h)
    delta=(4*h-h.sum(1,keepdims=True))/3 if 'delta' in condition else np.zeros_like(h)
    z=np.concatenate([h,interaction,delta,x],axis=2).reshape(n*4,-1)
    ah=z@p['w1']+p['b1'];hidden=np.maximum(ah,0)
    scores=(hidden@p['w2']+p['b2']).reshape(n,4)
    return scores,(body,question,a,h,aq,u,z,ah,hidden)


def backward(p,c,cache,ds):
    g={k:np.zeros_like(v) for k,v in p.items()};d=ds.reshape(-1,1)
    if c['package']=='P1':
        if c['capacity']=='linear':g['w']=cache[0].T@d;g['b']=d.sum(0);return g
        x,a,h=cache;g['w2']=h.T@d;g['b2']=d.sum(0)
        dh=(d@p['w2'].T)*(a>0);g['w1']=x.T@dh;g['b1']=dh.sum(0);return g
    body,question,a,h,aq,u,z,ah,hidden=cache
    n=len(ds);k=c.get('text_projection',32);condition=c['condition']
    g['w2']=hidden.T@d;g['b2']=d.sum(0)
    dh=(d@p['w2'].T)*(ah>0);g['w1']=z.T@dh;g['b1']=dh.sum(0)
    dz=(dh@p['w1'].T).reshape(n,4,-1)
    dbody=dz[:,:,:k].copy();du=np.zeros_like(u)
    if 'question' in condition:
        di=dz[:,:,k:2*k];dbody+=di*u[:,None,:];du=(di*h).sum(1)
    if 'delta' in condition:
        dd=dz[:,:,2*k:3*k];dbody+=(4*dd-dd.sum(1,keepdims=True))/3
    if condition!='format_final':
        da=(dbody*(a>0)).reshape(n*4,k)
        g['e']=np.asarray(body.T@da);g['eb']=da.sum(0)
        if 'question' in condition:
            dq=du*(aq>0);g['e']+=np.asarray(question.T@dq);g['eb']+=dq.sum(0)
    return g


class Adam:
    def __init__(self,p):
        self.m={k:np.zeros_like(v) for k,v in p.items()};self.v={k:np.zeros_like(v) for k,v in p.items()};self.t=0
    def step(self,p,g,c,masks):
        norm=np.sqrt(sum(float(np.square(v.astype(np.float64)).sum()) for v in g.values()))
        factor=min(1.,5/max(norm,1e-30));self.t+=1
        for k in p:
            grad=g[k]*factor
            if k not in ('b','b1','b2','eb'):grad=grad+c['l2']*p[k]*masks[k]
            self.m[k]*=.9;self.m[k]+=.1*grad
            self.v[k]*=.999;self.v[k]+=.001*grad*grad
            update=(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
            p[k]-=c['learning_rate']*update*masks[k]
        return float(norm)


def penalty(p,c,masks):
    return c['l2']/2*sum(float(np.square(v.astype(np.float64)*masks[k]).sum())
        for k,v in p.items() if k not in ('b','b1','b2','eb'))
