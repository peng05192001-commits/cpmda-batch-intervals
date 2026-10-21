from pathlib import Path
import os,sys,time,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
OLD=Path(__file__).resolve().parent
ROOT=OLD/'runs'
ROOT.mkdir(exist_ok=True)
sys.path.insert(0,str(OLD))
import benchmark as old
METHODS=['unsorted_chunk256','sorted_chunk1','sorted_chunk32','sorted_chunk256','implicit']
def main():
    target=ROOT/'supplemental_checks.json'
    if target.exists():raise RuntimeError('Existing result')
    out={'status':'running','seed':20261022,'scale':[],'stress':[],'model_fits':0,'query_targets_read':False}
    def save():target.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    save();rng=np.random.default_rng(20261022)
    out['initial_compile']=old.core.init_candidate()
    for n in [200,2000,20000]:
      for G in [6,200]:
       for B in [1,32,1000]:
        s,ptr,shifts,_=old.core.fixture(n,G,B)
        s=old.core.prepare('implicit',s,ptr,.9)
        for alpha in [.05,.1,.2]:
            p=(1-alpha)*(1+1/n);row={'n':n,'G':G,'B':B,'alpha':alpha,'timings':[],'max_error':0.}
            expected=old.core.numpy_endpoints('dense',s,ptr,shifts,p)
            for m in ['chunk32','implicit']:old.core.run(m,s,ptr,shifts,p)
            for rep in range(3):
                for m in rng.permutation(['chunk32','implicit']):
                    start=time.perf_counter();actual=old.core.run(m,s,ptr,shifts,p);elapsed=time.perf_counter()-start
                    np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-10)
                    row['max_error']=max(row['max_error'],float(np.max(np.abs(actual-expected))))
                    row['timings'].append({'method':str(m),'rep':rep,'seconds':elapsed})
            out['scale'].append(row);save()
       print('scale',n,G,'complete',flush=True)
    import joblib
    support=json.loads((OLD/'support.json').read_text(encoding='utf-8'))
    for rec in support['records']:
        with np.load(OLD/rec['input_file']) as d:xc,yc,xt=d['x_cal'].copy(),d['y_cal'],d['x_query'].copy()
        xc[:,:2]=np.nan;xt[:,:2]=np.nan
        state=joblib.load(OLD/rec['frozen_model_file'])
        def predict(x):return np.sort(np.column_stack([m.predict(state['prep'].transform(x)) for m in state['models']]),axis=1)
        raw,setup=old.prepare_cache('chunk256',xc,yc,xt,predict)
        ordered={k:(u,old.core.prepare('implicit',s,p,.9),p) for k,(u,s,p) in raw.items()}
        p=.9*(1+1/len(xc));expected,_=old.query('chunk32',xt,ordered,p,predict)
        row={'station':rec['station'],'masked_columns':['PM10','SO2'],'cal_missing_fraction':float(np.isnan(xc).mean()),'query_missing_fraction':float(np.isnan(xt).mean()),'patterns':len(raw),'G':[len(t[0]) for t in raw.values()],'setup':setup,'timings':[],'max_error':0.}
        for label in METHODS:
            old.query(label if label=='implicit' else label.split('_',1)[1],xt,raw if label.startswith('unsorted') else ordered,p,predict)
        for rep in range(5):
            for label in rng.permutation(METHODS):
                method=label if label=='implicit' else label.split('_',1)[1]
                actual,t=old.query(method,xt,raw if label.startswith('unsorted') else ordered,p,predict)
                np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-10)
                row['max_error']=max(row['max_error'],float(np.max(np.abs(actual-expected))))
                row['timings'].append({'method':str(label),'rep':rep,**t})
            print('stress',rec['station'],rep+1,'/5',flush=True)
        out['stress'].append(row);save()
    out['status']='complete';save()
if __name__=='__main__':main()
