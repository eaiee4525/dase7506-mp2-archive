"""Offline Train/synthetic engineering checks; no training or Validation/Test execution."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT'):
    os.environ[key]='1'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
BOOT="import runpy,sys; from pathlib import Path; p=Path(sys.argv.pop(1)).resolve(); sys.path.insert(0,str(p.parent)); runpy.run_path(str(p),run_name='__main__')"

def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);out=a.output.resolve()
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['PYTHONDONTWRITEBYTECODE']='1'
    def run(name,script,args):
        command=[sys.executable,'-I','-B','-c',BOOT,str(ROOT/script),*map(str,args)]
        start=time.perf_counter();p=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
        seconds=time.perf_counter()-start
        with (out/(name+'.stdout.txt')).open('x') as f:f.write(p.stdout)
        with (out/(name+'.stderr.txt')).open('x') as f:f.write(p.stderr)
        if p.returncode:raise RuntimeError(name+' failed: '+p.stderr)
        return seconds
    run('interface','verify_interface.py',['--data-dir',a.data_dir.resolve(),'--output',out/'interface.json'])
    # Newly generated arbitrary IDs/text/embeddings; deliberately no Reference file.
    audit=out/'synthetic_audit';(audit/'candidates').mkdir(parents=True)
    rows=[]
    for i in range(9):
        texts=['independent calculation #### 1/2',r'equivalent \boxed{0.5}','alternative #### -2','unparseable']
        if i==0:texts=['unknown']*4
        if i==1:texts=['#### 3']*4
        rows.append({'id':'new-audit-'+str(i),'question':'Unseen synthetic arithmetic input '+str(i),
                     'candidates':[{'index':j,'text':t} for j,t in enumerate(texts)]})
    with (audit/'candidates/gsm8k_audit.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    with (audit/'candidates/gsm8k_audit.npz').open('xb') as f:
        np.savez(f,embeddings=np.random.default_rng(20261030).normal(size=(36,1024)).astype(np.float32))
    files=('candidates/gsm8k_audit.jsonl','candidates/gsm8k_audit.npz')
    write(audit/'manifest.json',{'protocol':'dase7506-mp2-open-hard-verifier-v5','splits':{'audit':9},'sha256':{n:sha(audit/n) for n in files}})
    run('synthetic_audit','official/predict.py',['--method',ROOT/'submission.py','--checkpoint',ROOT/'assets',
        '--split','audit','--data-dir',audit,'--output',out/'audit_predictions.json'])
    audit_pred=json.loads((out/'audit_predictions.json').read_text());assert len(audit_pred['selected'])==9
    assert audit_pred['selected'][0:2]==[0,0]
    pred=run('train_prediction','official/predict.py',['--method',ROOT/'submission.py','--checkpoint',ROOT/'assets',
        '--split','train','--data-dir',a.data_dir.resolve(),'--output',out/'train_predictions.json'])
    score=run('train_scoring','official/score_predictions.py',['--split','train','--data-dir',a.data_dir.resolve(),
        '--predictions',out/'train_predictions.json','--output',out/'train_score.json'])
    disk=expanded=0
    for p in (ROOT/'assets').iterdir():
        disk+=p.stat().st_size
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as z:expanded+=sum(x.file_size for x in z.infolist())
        else:expanded+=p.stat().st_size
    assert expanded==61839 and expanded<=16*2**20
    poolcode="import numpy,scipy,threadpoolctl,json; print(json.dumps(threadpoolctl.threadpool_info()))"
    pool=subprocess.run([sys.executable,'-I','-B','-c',poolcode],env=env,capture_output=True,text=True,check=True)
    pools=json.loads(pool.stdout);assert all(p['num_threads']<=1 for p in pools)
    cpu=subprocess.run(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True) if sys.platform=='darwin' else None
    result={'python':platform.python_version(),'platform':platform.platform(),'architecture':platform.machine(),
        'cpu':cpu.stdout.strip() if cpu and cpu.returncode==0 else platform.processor(),
        'logical_cpus':os.cpu_count(),'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','threadpoolctl')},
        'fresh_processes_isolated_python':True,'data_scope':'Train and new synthetic audit only',
        'no_training':True,'no_validation_test_execution':True,'synthetic_audit_without_reference_passed':True,
        'asset_file_bytes':disk,'asset_uncompressed_bytes_including_headers_config_manifest':expanded,
        'prediction_process_seconds':pred,'independent_score_process_seconds':score,'total_seconds':pred+score,
        'timing_scope':'Full 4755-question Train input bank; startup/imports/data+asset loading/features/all experts and separate scoring included. Train resubstitution, not generalization.',
        'numeric_thread_limit':1,'sequential_subprocesses':True,
        'threadpools':[{k:p[k] for k in ('user_api','internal_api','num_threads','version','architecture') if k in p} for p in pools],
        'fresh_pip_install_performed':False,'reference_xeon_measured':False,
        'thread_evidence_limit':'Thread environment fixed to 1. An empty threadpools list means no registered runtime was reported, not that OS threads were measured here. Historical records/resource.json contains sampled OS-thread evidence; sampling is not continuous proof.'}
    write(out/'resource_summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
