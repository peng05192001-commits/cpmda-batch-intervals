from pathlib import Path
import json,sys,hashlib
import numpy as np
import joblib
import benchmark as b
def main():
 root=Path(__file__).resolve().parent
 b.core.correctness()
 result={'correctness':json.loads((root/'correctness.json').read_text()),'real':[]}
 support=json.loads((root/'support.json').read_text())
 for rec in support['records']:
  st=rec['station'];source=root/rec['input_file'];model=root/rec['frozen_model_file']
  assert b.sha(source)==rec['input_sha256'] and b.sha(model)==rec['frozen_model_sha256']
  with np.load(source) as d:xc,yc,xt=d['x_cal'],d['y_cal'],d['x_query']
  state=joblib.load(model)
  def predict(x):return np.sort(np.column_stack([m.predict(state['prep'].transform(x)) for m in state['models']]),axis=1)
  raw,_=b.prepare_cache('chunk256',xc,yc,xt,predict)
  ordered={k:(u,b.core.prepare('implicit',s,p,.9),p) for k,(u,s,p) in raw.items()}
  expected=None
  for label in ['unsorted_chunk256','sorted_chunk1','sorted_chunk32','sorted_chunk256','implicit']:
   m=label if label=='implicit' else label.split('_',1)[1]
   actual,_=b.query(m,xt,raw if label.startswith('unsorted') else ordered,.9*(1+1/len(xc)),predict)
   if expected is None:expected=actual
   assert np.array_equal(actual,expected)
   result['real'].append({'station':st,'method':label,'exact_equal':True,'endpoint_sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
   print(st,label,'exact match',flush=True)
 (root/'verification_local.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
