"""Frozen Phase14 probability mixture; only supplied inputs and model assets."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from .schema import Question
from .format_features import format_final
from . import decision as D
from .direct import group_logits

def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def base_raw(e,fn,w):
    if 'w' in w:
        return (((e-w['embedding_mean'])/w['embedding_scale']).reshape(-1,1024)@w['w']+w['b']).reshape(-1,4)
    x=(fn-w['fn_mean'])/w['fn_scale']
    z=np.concatenate([np.zeros((*x.shape[:2],96),np.float32),x],axis=2).reshape(-1,116)
    return (np.maximum(z@w['w1']+w['b1'],0)@w['w2']+w['b2']).reshape(-1,4)

def probability(s,g):
    mask=g.sum(2)>0;z=np.where(mask,s,-np.inf);z[~mask.any(1)]=0
    return softmax(z,axis=1)

def probabilities(bank,arrays,states):
    q=tuple(Question.from_bank(z) for z in bank);g,v=D.group_arrays(q)
    e=np.asarray(arrays['embeddings'],np.float32);fn,_=format_final(q)
    raw={k:base_raw(e,fn,states[k]) for k in ('B','C','F')};cal=states['decision']
    bf=D.group_features(raw['B'],raw['F'],g,v,cal['b_sd'],cal['b_rms'],'lme')@cal['b_w']
    cf=D.group_features(raw['C'],raw['F'],g,v,cal['c_sd'],cal['c_rms'],'lme')@cal['c_w']
    pre=states['preprocess']
    x=np.concatenate([(e-pre['em'])/pre['esd'],(fn-pre['fm'])/pre['fsd']],axis=2).astype(float)
    direct,*_=group_logits(states['direct']['theta'],x,g,'lme')
    pb,pc,pd=probability(bf,g),probability(cf,g),probability(direct,g)
    # Original arithmetic association; average probabilities, never raw scores.
    result=((pb+pc)/2+pd)/2
    return result,q,g,{'BF':pb,'CF':pc,'Direct_LME':pd}

def predict(bank,arrays,checkpoint=None):
    if not bank:return []
    e=np.asarray(arrays['embeddings'])
    if e.shape!=(len(bank),4,1024) or not np.isfinite(e).all():raise ValueError('Finite [N,4,1024] embeddings required')
    cp=Path(checkpoint) if checkpoint is not None else Path(__file__).resolve().parents[1]/'assets'
    cfg=json.loads((cp/'config.json').read_text())
    if cfg['version']!='phase14-full-train-v1' or cfg['weights']!={'BF':.25,'CF':.25,'Direct_LME':.5}:
        raise ValueError('Wrong frozen Phase14 configuration')
    with threadpool_limits(1):
        states={k:load(cp/(k+'.npz')) for k in ('B','C','F','decision','preprocess','direct')}
        p,q,g,_=probabilities(bank,{'embeddings':e},states);selected=D.choose(q,p,g)
    if len(selected)!=len(bank) or any(type(i) is not int or not 0<=i<4 for i in selected):raise ValueError('Invalid output')
    return selected
