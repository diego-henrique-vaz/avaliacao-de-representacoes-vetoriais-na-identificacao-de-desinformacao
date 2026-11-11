from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
TG=Path(os.environ.get('TG_RAIZ','/content/TG_REPRODUCAO_ROBUSTEZ'));R=TG/'robustez_privada_2026_11_06'
def load_data(base):
    d=pd.read_csv(R/'dados'/(base+'.csv'),keep_default_na=False,dtype={'doc_id':str,'pair_id':str,'group_id':str})
    if base=='fakebr' and (R/'dados/GRUPOS_REFORCADOS_FAKEBR.csv').exists():
        m=pd.read_csv(R/'dados/GRUPOS_REFORCADOS_FAKEBR.csv',dtype=str);assert m.doc_id.tolist()==d.doc_id.tolist() and m.group_id_original.tolist()==d.group_id.tolist()
        d['group_id_original']=d.group_id;d['split_original']=d.split;d['group_id']=m.group_id_robust
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(d,groups=d.group_id));d['split']='train';d.loc[te,'split']='test'
    return d
