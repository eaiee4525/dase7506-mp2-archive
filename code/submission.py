"""Official offline entrypoint for the fixed Phase14 full-Train candidate."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
             'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT'):
    os.environ[_key]='1'
from mp2_model.inference import predict
__all__=['predict']
