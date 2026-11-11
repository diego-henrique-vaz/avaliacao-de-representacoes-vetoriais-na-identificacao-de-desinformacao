"""Experimentos adicionais, sem alterar o notebook ou os resultados históricos."""
from pathlib import Path
import os
import sys,json,re,unicodedata,hashlib,time,gc,platform,importlib.metadata,threading
WORK=Path(__file__).resolve().parent
if (WORK/'testdeps').exists():sys.path.insert(0,str(WORK/'testdeps'))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
from gensim.models import Word2Vec,FastText
from scipy.sparse import save_npz,issparse
from xgboost import XGBClassifier
import psutil
TG=Path(os.environ.get('TG_RAIZ', '/content/TG_REPRODUCAO_ROBUSTEZ'));R=TG/'experimentos_divisoes_2026_11_05';N=TG/'revisao_desenvolvimento_2026_11_03/notebooks'
METHODS=['tfidf','word2vec','fasttext','bertimbau','e5','qwen3'];BASES=['fakebr','boatosbr','faketrue','pteu']
PARAMS=dict(objective='binary:logistic',tree_method='hist',n_estimators=300,max_depth=3,learning_rate=.05,subsample=1.,colsample_bytree=1.,random_state=42,n_jobs=2)

def dump(path,value):
    # Temporário exclusivo evita colisões de inicialização entre workers disjuntos.
    import uuid
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(path)
    finally:
        if tmp.exists():tmp.unlink()

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tokenizar(texto):return re.findall(r'(?u)\b\w\w+\b',unicodedata.normalize('NFC',texto).lower())
def hash_estavel(palavra):return int.from_bytes(hashlib.sha256(palavra.encode('utf8')).digest()[:8],'little')
def vetor_documento(palavras,modelo,sub=False):
    v=[modelo.wv[p] for p in palavras if sub or p in modelo.wv.key_to_index]
    return np.mean(v,axis=0) if v else np.zeros(100)
def fingerprint(d):return hashlib.sha256(json.dumps(d[['doc_id','pair_id','group_id','label','text','split']].to_dict('records'),sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def intervalos(teste,probabilidades,baseline=None):
    """Bootstrap dos grupos; matriz de contagens evita recalcular F1 por texto."""
    y=teste.label.to_numpy();g,unique=pd.factorize(teste.group_id,sort=True);G=len(unique)
    rng=np.random.default_rng(42);weights=np.empty((2000,G),dtype=np.int32)
    for i in range(2000):weights[i]=np.bincount(rng.integers(G,size=G),minlength=G)
    def samples(pred):
        cells=np.zeros((G,4),dtype=np.int32);np.add.at(cells,(g,y*2+pred),1)
        c=weights@cells;tn,fp,fn,tp=c.T
        pos=2*tp/np.maximum(2*tp+fp+fn,1);neg=2*tn/np.maximum(2*tn+fp+fn,1)
        valid=((tn+fp)>0)&((fn+tp)>0)
        return (pos+neg)/2,valid
    scores={};valids={}
    for name,prob in probabilidades.items():scores[name],valids[name]=samples((np.asarray(prob)>=.5).astype(int))
    if baseline:
        for name,pred in baseline.items():scores['historico_'+name],valids['historico_'+name]=samples(np.asarray(pred,dtype=int))
    out={'repeticoes_solicitadas':2000,'amostragem':'grupos inteiros com reposição, semente 42; classificadores fixos','intervalos':{},'diferencas_pareadas_exploratorias':[]}
    for name in probabilidades:
        vals=scores[name][valids[name]];out['intervalos'][name]={'lower':float(np.quantile(vals,.025)),'upper':float(np.quantile(vals,.975)),'valid_repetitions':int(len(vals))}
        if baseline:
            valid=valids[name]&valids['historico_'+name];diff=scores[name][valid]-scores['historico_'+name][valid]
            out['diferencas_pareadas_exploratorias'].append({'method':name,'contrast':'novo menos histórico80/20 no teste comum','lower':float(np.quantile(diff,.025)),'upper':float(np.quantile(diff,.975))})
    return out

def extrair(d,tr,te,metodo):
    texts=d.text.tolist();Xtr=[texts[i] for i in tr];Xte=[texts[i] for i in te]
    if metodo=='tfidf':
        v=TfidfVectorizer(tokenizer=tokenizar,token_pattern=None,preprocessor=None,lowercase=False,max_features=5000,ngram_range=(1,2),max_df=.85,min_df=2,dtype=np.float32)
        return v.fit_transform(Xtr),v.transform(Xte),{'vocabulary_size':len(v.vocabulary_)}
    pt=[tokenizar(t) for t in Xtr];pe=[tokenizar(t) for t in Xte]
    common=dict(sentences=pt,vector_size=100,window=5,min_count=1,workers=1,sg=1,seed=42,epochs=10,negative=5,hs=0,sample=.001,ns_exponent=.75,alpha=.025,min_alpha=.0001,hashfxn=hash_estavel)
    model=FastText(**common,min_n=3,max_n=6,bucket=200000) if metodo=='fasttext' else Word2Vec(**common)
    a=np.asarray([vetor_documento(p,model,metodo=='fasttext') for p in pt],dtype=np.float32)
    b=np.asarray([vetor_documento(p,model,metodo=='fasttext') for p in pe],dtype=np.float32)
    coverage={}
    for split,words in [('train',pt),('test',pe)]:
        total=sum(map(len,words));known=sum(p in model.wv.key_to_index for doc in words for p in doc)
        coverage[split]={'tokens':total,'known_tokens':known,'oov_tokens':total-known,'known_fraction':known/max(total,1),'zero_vectors':int(np.sum(~np.any(a if split=='train' else b,axis=1)))}
    del model;gc.collect();return a,b,{'vocabulary_coverage':coverage}

def main():
    R.joinpath('evidencias').mkdir(parents=True,exist_ok=True)
    origins=json.loads((R/'dados/ORIGEM_E_PARTICOES.json').read_text(encoding='utf8'))
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','gensim','xgboost','psutil']};versions['python']=platform.python_version();versions['platform']=platform.platform()
    dump(R/'evidencias/VERSOES_NOVOS_TESTES.json',versions)
    status={'status':'EM_ANDAMENTO','planned_new_fits':48,'completed_new_fits':0,'old_fraction_groups':[.8,.2],'new_fractions_groups':[[.5,.5],[.3,.7]],'split_seed':42,'train_seed':42,'xgboost':PARAMS,'new_encoder_extractions':0,'datasets':{}}
    allrows=[];commonrows=[];cache_records=[];splitrows=[]
    dump(R/'evidencias/ANDAMENTO.json',status)
    for base in BASES:
        path=R/'dados'/(base+'.csv');assert sha(path)==origins[base]['csv_sha256']
        d=pd.read_csv(path,keep_default_na=False,dtype={'doc_id':str,'pair_id':str,'group_id':str})
        fp=fingerprint(d);assert fp==origins[base]['original_data_manifest']['fingerprint']
        original_train=np.flatnonzero(d.split.eq('train'));original_test=np.flatnonzero(d.split.eq('test'))
        historical=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(d,groups=d.group_id))
        assert np.array_equal(historical[0],original_train) and np.array_equal(historical[1],original_test)
        caches={};refs={}
        for method in METHODS:
            refs[method]=pd.read_csv(N/'referencias_resultados'/base/(method+'.csv'),dtype={'doc_id':str})
            assert refs[method].doc_id.tolist()==d.iloc[original_test].doc_id.tolist()
            allrows.append({'base':base,'train_fraction_groups':.8,'test_fraction_groups':.2,'method':method,'n_train':len(original_train),'n_test':len(original_test),'macro_f1':float(f1_score(refs[method].label,refs[method].prediction,average='macro')),'accuracy':float(accuracy_score(refs[method].label,refs[method].prediction)),'origin':'historical_80_20_no_new_fit'})
            if method not in ['bertimbau','e5','qwen3']:continue
            p=R/'vetores_contextuais'/base/method;spec=json.loads((p/'spec.json').read_text(encoding='utf8'));progress=json.loads((p/'progress.json').read_text(encoding='utf8'))
            assert spec['fingerprint']==fp and spec['name']==method and spec['max_length']==512
            assert progress['next_index']==len(d)
            a=np.load(p/'all.npy',mmap_mode='r',allow_pickle=False)
            assert a.shape==(len(d),spec['model']['dim']) and a.dtype==np.float32 and np.isfinite(a).all()
            historical_model=XGBClassifier();historical_model.load_model(p/'xgboost.json');prob=historical_model.predict_proba(a[original_test])[:,1]
            maxdiff=float(np.max(np.abs(prob-refs[method].prob_fake.to_numpy())))
            assert np.array_equal((prob>=.5).astype(int),refs[method].prediction.to_numpy()) and maxdiff<1e-6
            cache_records.append({'base':base,'method':method,'all_sha256':sha(p/'all.npy'),'bytes':(p/'all.npy').stat().st_size,'shape':list(a.shape),'fingerprint':fp,'model':spec['model'],'precision':spec['precision'],'baseline_prediction_check':'PASS','max_probability_difference':maxdiff})
            caches[method]=a;del historical_model
        dump(R/'evidencias/VALIDACAO_CACHES.json',cache_records)
        print('CACHE VALIDADO',base,'documentos',len(d),flush=True)
        status['datasets'][base]={'rows':len(d),'groups':int(d.group_id.nunique()),'historical_train':len(original_train),'historical_test':len(original_test),'conditions':{}}
        for fraction in [.5,.7]:
            tag='50_50' if fraction==.5 else '30_70';out=R/'resultados'/base/tag;out.mkdir(parents=True,exist_ok=True)
            tr,te=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=42).split(d,groups=d.group_id));train=d.iloc[tr];test=d.iloc[te]
            assert len(set(train.group_id)&set(test.group_id))==0
            assert len(set(train.canonical_sha256)&set(test.canonical_sha256))==0
            assert train.label.nunique()==2 and test.label.nunique()==2
            assert set(tr)<=set(original_train) and set(original_test)<=set(te)
            cfg={'base':base,'source_csv_sha256':sha(path),'fingerprint_original_order':fp,'test_fraction_groups':fraction,'split_seed':42,'train_seed':42,'params_xgboost':PARAMS,'train_ids':train.doc_id.tolist(),'test_ids':test.doc_id.tolist(),'all_texts_unchanged':True}
            cfgsig=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest();dump(out/'CONFIGURACAO.json',cfg)
            part=d[['doc_id','pair_id','group_id','canonical_sha256','label']].copy();part['split']='train';part.loc[te,'split']='test';part.to_csv(out/'particoes.csv',index=False)
            splitrow={'base':base,'condition':tag,'train_fraction_groups':1-fraction,'test_fraction_groups':fraction,'train_groups':int(train.group_id.nunique()),'test_groups':int(test.group_id.nunique()),'train_docs':len(tr),'test_docs':len(te),'train_true':int(sum(train.label==0)),'train_false':int(sum(train.label==1)),'test_true':int(sum(test.label==0)),'test_false':int(sum(test.label==1)),'common_test_docs':len(original_test),'groups_disjoint':True,'canonical_hashes_disjoint':True,'nested_in_historical_train':True}
            splitrows.append(splitrow);status['datasets'][base]['conditions'][tag]=splitrow
            pd.DataFrame(splitrows).to_csv(R/'DIVISOES_E_CONTAGENS.csv',index=False)
            probs={};common_probs={};common_baseline={}
            for method in METHODS:
                resultpath=out/(method+'_resultado.json');predictionpath=out/(method+'_predicoes.csv')
                if resultpath.exists():
                    row=json.loads(resultpath.read_text(encoding='utf8'));assert row['configuration_signature']==cfgsig
                    saved=pd.read_csv(predictionpath,dtype={'doc_id':str});assert saved.doc_id.tolist()==test.doc_id.tolist()
                    assert sha(predictionpath)==row['predictions_sha256'];prob=saved.prob_fake.to_numpy()
                else:
                    print('INICIANDO',base,tag,method,flush=True)
                    t=time.perf_counter();peak=[psutil.Process().memory_info().rss];stop=threading.Event()
                    def monitor():
                        while not stop.wait(.1):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
                    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
                    try:
                        if method in caches:Xtr,Xte,extra=caches[method][tr],caches[method][te],{'frozen_cache_sha256':next(x['all_sha256'] for x in cache_records if x['base']==base and x['method']==method)}
                        else:Xtr,Xte,extra=extrair(d,tr,te,method)
                        extraction_seconds=time.perf_counter()-t
                        model=XGBClassifier(**PARAMS);t=time.perf_counter();model.fit(Xtr,train.label.to_numpy());fit_seconds=time.perf_counter()-t
                        prob=model.predict_proba(Xte)[:,1];pred=(prob>=.5).astype(int)
                        model.save_model(out/(method+'_xgboost.json'))
                        if method not in caches:
                            p=out/method;p.mkdir(exist_ok=True)
                            if issparse(Xtr):save_npz(p/'train.npz',Xtr);save_npz(p/'test.npz',Xte)
                            else:np.save(p/'train.npy',Xtr);np.save(p/'test.npy',Xte)
                        saved=test[['doc_id','group_id','label']].copy();saved['prob_fake']=prob;saved['prediction']=pred;saved.to_csv(predictionpath,index=False)
                        report=classification_report(test.label,pred,labels=[0,1],target_names=['Verdadeira','Falsa'],output_dict=True,zero_division=0)
                        row={'base':base,'train_fraction_groups':1-fraction,'test_fraction_groups':fraction,'method':method,'n_train':len(tr),'n_test':len(te),'macro_f1':float(f1_score(test.label,pred,average='macro')),'accuracy':float(accuracy_score(test.label,pred)),'classification_report':report,'confusion_matrix':confusion_matrix(test.label,pred,labels=[0,1]).tolist(),'representation_seconds_local':extraction_seconds,'fit_seconds_local':fit_seconds,'peak_process_rss_mib_sampled':peak[0]/(1024**2),'origin':'new_fit_frozen_cache' if method in caches else 'new_representation_and_new_fit','configuration_signature':cfgsig,'predictions_sha256':sha(predictionpath),**extra}
                        dump(resultpath,row);del model,Xtr,Xte;gc.collect()
                    finally:stop.set();watcher.join()
                probs[method]=prob
                byid=pd.Series(prob,index=test.doc_id);common_probs[method]=byid.loc[d.iloc[original_test].doc_id].to_numpy();common_baseline[method]=refs[method].prediction.to_numpy()
                common_pred=(common_probs[method]>=.5).astype(int);metric=float(f1_score(d.iloc[original_test].label,common_pred,average='macro'))
                commonrows.append({'base':base,'train_fraction_groups':1-fraction,'test_fraction_groups':fraction,'method':method,'common_test_docs':len(original_test),'macro_f1_common':metric,'historical_macro_f1_same_test':float(f1_score(refs[method].label,refs[method].prediction,average='macro')),'delta_vs_historical':metric-float(f1_score(refs[method].label,refs[method].prediction,average='macro'))})
                allrows.append(row);status['completed_new_fits']+=1
                pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in allrows]).to_csv(R/'COMPARACAO_PROPORCOES.csv',index=False)
                pd.DataFrame(commonrows).to_csv(R/'COMPARACAO_TESTE_COMUM.csv',index=False);dump(R/'evidencias/ANDAMENTO.json',status)
                print('CONCLUIDO',base,tag,method,'macroF1',round(row['macro_f1'],4),'comum',round(metric,4),status['completed_new_fits'],'/48',flush=True)
            dump(out/'bootstrap_teste_completo.json',intervalos(test,probs))
            dump(out/'bootstrap_teste_comum.json',intervalos(d.iloc[original_test],common_probs,common_baseline))
            dump(R/'evidencias/ANDAMENTO.json',status)
        del caches;gc.collect()
    assert status['completed_new_fits']==48 and len(allrows)==72 and len(commonrows)==48
    status['status']='CONCLUIDO';status['reported_rows_including_historical']=72;status['common_test_rows']=48;dump(R/'evidencias/ANDAMENTO.json',status)
    print('FINALIZADO:48 novos treinos;24 resultados históricos preservados.',flush=True)

if __name__=='__main__':main()
