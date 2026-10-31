"""Train-input and synthetic-only checks; refuses any evaluation data dependency."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_THREAD_LIMIT'):
    os.environ[k]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import socket
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from official.common import load_inputs,validate_choices
from mp2_model import inference

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,default=Path(__file__).with_name('assets'))
    args=ap.parse_args();bank,arr=load_inputs('train',args.data_dir)
    # Warm a standard-library lazy import before restricting actual data-file access.
    with threadpool_limits(1): pass
    cp=args.checkpoint.resolve();opened=[];guard_active=[False]
    def hook(event,values):
        if not guard_active[0]:return
        if event.startswith('socket.'):raise AssertionError('Network during predict')
        if event=='open':
            p=values[0]
            if isinstance(p,(str,bytes,os.PathLike)):
                path=Path(p).resolve();opened.append(str(path.name))
                if not path.is_relative_to(cp):raise AssertionError('Inference opened non-asset path: '+str(path))
    sys.addaudithook(hook)
    def call(bb,aa,checkpoint=cp):
        guard_active[0]=True
        try:return inference.predict(bb,aa,checkpoint)
        finally:guard_active[0]=False
    result={};sizes=(0,1,7,32,467,798,len(bank));chosen={}
    for n in sizes:
        value=call(bank[:n],{'embeddings':arr['embeddings'][:n]})
        assert len(value)==n and all(type(i) is int and 0<=i<4 for i in value)
        chosen[n]=value
    result['native_int_output_shapes']=list(sizes)
    original=chosen[len(bank)]
    assert call(bank,arr)==original;result['repeat_exact']=True
    default_cp=Path(__file__).with_name('assets').resolve()
    if cp==default_cp:
        assert call(bank[:32],{'embeddings':arr['embeddings'][:32]},None)==chosen[32]
        result['relative_default_checkpoint']=True
    else:result['relative_default_checkpoint']='not exercised for a nondefault checkpoint'
    rng=np.random.default_rng(20261029);order=rng.permutation(len(bank))
    reordered=call([bank[i] for i in order],{'embeddings':arr['embeddings'][order]})
    assert reordered==[original[i] for i in order];result['question_reordering_exact']=True
    anon=[{**row,'id':'synthetic-unseen-'+str(i)} for i,row in enumerate(bank)]
    assert call(anon,arr)==original;result['id_renaming_exact']=True
    chunk=[]
    for start in range(0,len(bank),127):chunk+=call(bank[start:start+127],{'embeddings':arr['embeddings'][start:start+127]})
    assert chunk==original;result['subbatch_selection_exact']=True
    # Entirely new text/embeddings: no labels and no lookup keys can be present.
    sim=[]
    for i in range(9):
        texts=[r'calculation #### 1/2',r'alternative \boxed{0.5}',r'calculation #### -2','unparseable']
        if i==0:texts=['unknown']*4
        if i==1:texts=['#### 3']*4
        sim.append({'id':'unseen-'+str(i),'question':'Synthetic arithmetic interface probe '+str(i),
                    'candidates':[{'index':j,'text':t} for j,t in enumerate(texts)]})
    sx=rng.normal(size=(9,4,1024)).astype(np.float32)
    out=call(sim,{'embeddings':sx});assert len(out)==9 and out[0]==0 and out[1]==0
    result['synthetic_unseen_duplicate_invalid_single_group']=True
    from mp2_model.decision import group_arrays,choose
    from mp2_model.schema import Question
    q=[Question.from_bank(sim[2])];g,v=group_arrays(q)
    assert choose(q,np.zeros((1,4)),g)==[0];result['exact_tie_earliest_candidate']=True
    validate_choices(bank,[r['id'] for r in bank],original)
    result.update({'no_network_or_non_asset_reads_in_actual_inference':True,
        'opened_asset_filenames':sorted(set(opened)),
        'train_prediction_sha256':hashlib.sha256(json.dumps(original).encode()).hexdigest(),
        'input_scope':'Train and synthetic only; no Validation/Test',
        'package_versions':{'numpy':np.__version__},'thread_configuration':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'isolated_source_modules':sorted(m for m in sys.modules if m.startswith('phase'))})
    assert result['isolated_source_modules']==[]
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
