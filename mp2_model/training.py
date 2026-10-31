import json,hashlib,time
from pathlib import Path
import numpy as np
from . import frozen_models as models
from .legacy_meta import meta_features
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p): return json.loads(Path(p).read_text())

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else list(x))+'\n'
    with p.open('x') as f:f.write(text)

def save(p,**v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**v)

def scaler(x):
    a=x.reshape(-1,x.shape[-1]).astype(np.float64);mean=a.mean(0);sd=a.std(0);sd[sd==0]=1
    return mean.astype(np.float32),sd.astype(np.float32)

def train_base(c,e,fn,y,g,v,d,fit_indices,eval_indices,out):
    start=time.perf_counter();p=models.initialize(c);masks=models.active_masks(p,c);opt=models.Adam(p)
    raw=e if c['package']=='P1' else fn
    mean,scale=scaler(raw[fit_indices]);x=(raw-mean)/scale
    pp={'embedding_mean':mean,'embedding_scale':scale} if c['package']=='P1' else {'fn_mean':mean,'fn_scale':scale}
    keep=fit_indices[d[fit_indices]] if c['population']=='disagreement' else fit_indices
    assert len(keep)>0 and not set(keep)&set(eval_indices)
    rng=np.random.default_rng(c['seed']+1);need=c['updates']*c['batch_questions'];n=len(keep)
    order=keep[np.concatenate([rng.permutation(n) for _ in range((need+n-1)//n)])[:need]]
    history=[];maxnorm=0.;clipped=0;started=time.perf_counter()
    def forward(ix,params):
        xx=x[ix].reshape(-1,1024) if c['package']=='P1' else x[ix]
        return models.forward(params,c,xx,None,None)
    for step in range(c['updates']):
        ix=order[step*c['batch_questions']:(step+1)*c['batch_questions']]
        s,cache=forward(ix,p);loss,ds=models.loss_grad(s,y[ix],v[ix],g[ix],c['objective'])
        grad=models.backward(p,c,cache,ds)
        if not np.isfinite(loss) or any(not np.isfinite(z).all() for z in grad.values()):raise FloatingPointError('Nonfinite fit')
        norm=opt.step(p,grad,c,masks);maxnorm=max(maxnorm,norm);clipped+=norm>5
        if step==0 or (step+1)%100==0:
            history.append({'step':step+1,'bce':loss,'l2_penalty':models.penalty(p,c,masks),'gradient_norm':norm})
    seconds=time.perf_counter()-started
    save(out/'model.npz',**p,**pp)
    # Full parameter checkpoint stays private. Export only arrays actually used at inference.
    needed=('w','b') if c['package']=='P1' else ('w1','b1','w2','b2')
    exported={**{k:p[k] for k in needed},**pp}
    save(out/'inference.npz',**exported)
    pred=forward(eval_indices,p)[0] if len(eval_indices) else np.empty((0,4),np.float32)
    probe=eval_indices if len(eval_indices) else fit_indices[:32]
    with np.load(out/'model.npz',allow_pickle=False) as f:reload={k:f[k] for k in p}
    assert np.array_equal(forward(probe,p)[0],forward(probe,reload)[0])
    if len(eval_indices):save(out/'heldout_scores.npz',indices=eval_indices,candidate_logits=pred)
    record={'config':c,'preprocessing_fit_indices':fit_indices.tolist(),'optimization_indices':keep.tolist(),
       'eval_indices':eval_indices.tolist(),'updates':opt.t,'fit_seconds':seconds,
       'preprocess_and_fit_seconds':time.perf_counter()-start,'history':history,
       'finite':all(np.isfinite(z).all() for z in p.values()),'max_gradient_norm':maxnorm,'clipped_steps':int(clipped),
       'question_exposures':len(order),'disagreement_exposures':int(d[order].sum()),
       'fit_population_questions':len(keep),'preprocessing_questions':len(fit_indices),
       'inference_array_bytes':sum(z.nbytes for z in exported.values()),
       'model_sha256':sha(out/'model.npz'),'inference_sha256':sha(out/'inference.npz'),'export_exact':True}
    write(out/'training.json',record)
    print(json.dumps({'trained':out.name,'population':len(keep),'scaler_population':len(fit_indices),
                      'fit_seconds':round(seconds,4)}),flush=True)
    return pred,exported,record

def normalizers(a,b,g,v,d):
    aa,bb,gg,vv=a[d],b[d],g[d],v[d];sd=[]
    for s in (aa,bb):
        ss=s-(s*vv).sum(1,keepdims=True)/vv.sum(1,keepdims=True)
        sd.append(max(float(np.sqrt((ss*ss*vv).sum()/vv.sum())),1e-4))
    # Historical RMS is fitted on MEAN candidate features, never re-estimated on LME.
    x=meta_features(aa,bb,gg,vv,sd)
    rms=np.maximum(np.sqrt((x*x*vv[:,:,None]).sum((0,1))/vv.sum()),1e-4)
    return sd,rms
