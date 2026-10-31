"""Reproduce the existing Phase13/14 full-Train protocol, without tuning."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
          'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT'):os.environ[k]='1'
import argparse,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from official.common import load_inputs,read_jsonl,parse_answer,check_data
from mp2_model.schema import Question,partition
from mp2_model.format_features import format_final
from mp2_model.training import read,write,save,train_base,normalizers,scaler
from mp2_model import decision as D,direct

def run(data_dir,output):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    cfg=read(Path(__file__).with_name('frozen_training.json'));check_data(data_dir,('train',))
    bank,arr=load_inputs('train',data_dir);q=[Question.from_bank(r) for r in bank];e=arr['embeddings']
    refs=read_jsonl(data_dir/'gsm8k_train.jsonl')
    assert [r['id'] for r in refs]==[z.id for z in q] and all(r['question']==z.question for r,z in zip(refs,q))
    gold=[parse_answer(r['answer']) for r in refs];assert None not in gold
    y=np.array([[a is not None and a==answer for a in z.answers] for z,answer in zip(q,gold)],bool)
    assert y.any(1).all()
    fn,_=format_final(q);g,v=D.group_arrays(q);d=np.array([z.disagreement for z in q]);ids=[z.id for z in q]
    buckets=partition(q,3,cfg['calibration_seed']);write(output/'membership.json',{'ids':ids,'heldout_ids':buckets})
    inside={k:np.full((len(q),4),np.nan,np.float32) for k in ('B','C','F')}
    assets=output/'assets';state={};recs={};assets.mkdir()
    with threadpool_limits(1):
        for j,bucket in enumerate(buckets):
            held=set(bucket);ti=np.array([i for i,z in enumerate(q) if z.id not in held]);vi=np.array([i for i,z in enumerate(q) if z.id in held])
            assert not {q[i].question_key for i in ti}&{q[i].question_key for i in vi}
            local=np.zeros_like(y);local[ti]=y[ti]
            for k in inside:
                s,_,_=train_base(cfg['configs'][k],e,fn,local,g,v,d,ti,vi,output/'inner'/str(j)/k);inside[k][vi]=s
        assert all(np.isfinite(s).all() for s in inside.values())
        save(output/'calibration_oof.npz',**inside)
        for k,prefix in [('B','b'),('C','c')]:
            sd,rms=normalizers(inside[k],inside['F'],g,v,d)
            w,rec=D.fit(inside[k],inside['F'],y,g,v,sd,rms,'lme','ce');assert rec['success'];recs[k]=rec
            state.update({prefix+'_w':w,prefix+'_sd':np.asarray(sd),prefix+'_rms':rms})
        save(assets/'decision.npz',**state)
        for k in inside:
            _,exported,_=train_base(cfg['configs'][k],e,fn,y,g,v,d,np.arange(len(q)),np.array([],int),output/'full'/k)
            save(assets/(k+'.npz'),**exported)
        em,esd=scaler(e);fm,fsd=scaler(fn);save(assets/'preprocess.npz',em=em,esd=esd,fm=fm,fsd=fsd)
        x=np.concatenate([(e-em)/esd,(fn-fm)/fsd],axis=2).astype(float)
        theta=np.r_[np.zeros(1044),1.]
        mean,meanrec=direct.fit(x[d],g[d],y[d],'mean',theta);assert meanrec['success']
        theta,lmerec=direct.fit(x[d],g[d],y[d],'lme',mean);assert lmerec['success']
        save(output/'direct_mean_initialization.npz',theta=mean);save(assets/'direct.npz',theta=theta)
    write(assets/'config.json',cfg['inference'])
    write(output/'training_complete.json',{'base_fits':12,'calibrator_fits':2,'direct_stages':{'mean':meanrec,'lme':lmerec},
         'calibrators':recs,'questions':len(q),'disagreement':int(d.sum()),'seconds':time.perf_counter()-start,
         'train_accuracy_computed':False,'external_evaluation_performed':False})
    print('Fixed full-Train protocol completed; no accuracy selection or external evaluation.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();run(a.data_dir,a.output)
