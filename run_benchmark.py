"""Predeclared revision check; uses frozen local inputs and original kernels."""
from pathlib import Path
import os,sys,time,json,hashlib,platform
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
OLD=Path(__file__).resolve().parent
ROOT=OLD/'runs'
ROOT.mkdir(exist_ok=True)
sys.path.insert(0,str(OLD))
import benchmark as old
METHODS=['unsorted_chunk256','sorted_chunk1','sorted_chunk32','sorted_chunk256','implicit']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(obj):
    (ROOT/'repeat_timing.json').write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def main():
    if (ROOT/'repeat_timing.json').exists(): raise RuntimeError('Existing run: inspect, do not overwrite')
    import ctypes,joblib,sklearn
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    handle=kernel.GetCurrentProcess()
    process_mask,system_mask=ctypes.c_size_t(),ctypes.c_size_t()
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    assert kernel.GetProcessAffinityMask(handle,ctypes.byref(process_mask),ctypes.byref(system_mask))
    first_mask=process_mask.value & -process_mask.value
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    assert kernel.SetProcessAffinityMask(handle,first_mask)
    class Mem(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ['total','avail','page','availpage','virt','availvirt','ext']]
    mem=Mem();mem.length=ctypes.sizeof(mem);assert kernel.GlobalMemoryStatusEx(ctypes.byref(mem))
    rng=np.random.default_rng(20261020)
    support=json.loads((OLD/'support.json').read_text(encoding='utf-8'))
    paths=[Path(__file__),OLD/'benchmark_protocol.md',OLD/'benchmark.py',OLD/'selector.py',OLD/'experiment.py',OLD/'support.json']
    out={'status':'running','seed':20261020,'repeats':20,'python':sys.version,'numpy':np.__version__,
         'sklearn':sklearn.__version__,'platform':platform.platform(),'cpu':platform.processor(),
         'cpu_affinity_mask':first_mask,'physical_memory_bytes':mem.total,
         'sources':{str(p):sha(p) for p in paths},'stations':[],'model_fits':0,'query_targets_read':False}
    out['initial_compile']=old.core.init_candidate();save(out)
    for rec in support['records']:
        station=rec['station'];source=OLD/rec['input_file'];model=OLD/rec['frozen_model_file']
        assert sha(source)==rec['input_sha256'] and sha(model)==rec['frozen_model_sha256']
        with np.load(source) as d: xc,yc,xt=d['x_cal'],d['y_cal'],d['x_query']
        state=joblib.load(model)
        def predict(x):
            xx=state['prep'].transform(x)
            return np.sort(np.column_stack([m.predict(xx) for m in state['models']]),axis=1)
        raw,setup=old.prepare_cache('chunk256',xc,yc,xt,predict)
        sorted_cache={};begin=time.perf_counter()
        for key,(unions,scores,ptr) in raw.items():
            sorted_cache[key]=(unions,old.core.prepare('implicit',scores,ptr,.9*(1+1/len(xc))),ptr)
        sorting=time.perf_counter()-begin
        with np.load(OLD/'runs'/f'{station}_implicit.npz') as d: expected=d['endpoints']
        r={'station':station,'n':len(xc),'B':len(xt),'setup':setup,'additional_sort_seconds':sorting,
           'calibration_missing_rates':np.isnan(xc).mean(axis=0).tolist(),
           'query_missing_rates':np.isnan(xt).mean(axis=0).tolist(),
           'pattern_details':rec['query_patterns'],
           'sorted_score_bytes':sum(s.nbytes for _,s,_ in sorted_cache.values()),
           'model_parameters':[m.get_params() for m in state['models']],
           'timings':[],'orders':[],'max_endpoint_error':0.,'all_exact_equal':True}
        out['stations'].append(r)
        def call(label,rep):
            method=label if label=='implicit' else label.split('_',1)[1]
            actual,t=old.query(method,xt,raw if label.startswith('unsorted') else sorted_cache,.9*(1+1/len(xc)),predict)
            np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-10)
            r['max_endpoint_error']=max(r['max_endpoint_error'],float(np.max(np.abs(actual-expected))))
            r['all_exact_equal'] &= bool(np.array_equal(actual,expected))
            if rep>=0:r['timings'].append({'method':label,'repetition':rep,**t})
        print(station,'warming',flush=True)
        for label in METHODS:call(label,-1)
        for rep in range(20):
            order=[METHODS[i] for i in rng.permutation(5)];r['orders'].append(order)
            for label in order:call(label,rep)
            save(out);print(station,'block',rep+1,'/20',flush=True)
        values={m:np.array([t['total_seconds'] for t in r['timings'] if t['method']==m]) for m in METHODS}
        r['summary']={m:{'median':float(np.median(v)),'min':float(v.min()),'max':float(v.max())} for m,v in values.items()}
        brng=np.random.default_rng(20261021);idx=brng.integers(0,20,(5000,20));r['comparisons']={}
        for m in METHODS[:-1]:
            b,c=values[m],values['implicit']
            reductions=100*(1-np.median(c[idx],axis=1)/np.median(b[idx],axis=1))
            diffs=np.median((b-c)[idx],axis=1)
            r['comparisons'][m]={'reduction_percent':float(100*(1-np.median(c)/np.median(b))),
              'reduction_ci95':np.quantile(reductions,[.025,.975]).tolist(),
              'paired_median_seconds_saved':float(np.median(b-c)),
              'paired_difference_ci95':np.quantile(diffs,[.025,.975]).tolist()}
        save(out)
    from threadpoolctl import threadpool_info
    out.update(status='complete',threadpools=threadpool_info(),sources_unchanged=all(sha(Path(p))==h for p,h in out['sources'].items()))
    save(out)
if __name__=='__main__':main()
