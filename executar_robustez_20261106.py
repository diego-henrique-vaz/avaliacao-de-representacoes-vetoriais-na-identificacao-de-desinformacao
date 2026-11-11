"""Cinco divisões por grupos, controle de comprimento e matrizes de transferência."""
from pathlib import Path
import sys,json,hashlib,time,gc,platform,importlib.metadata,traceback,os
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'testdeps'))
import numpy as np
import pandas as pd
from scipy.sparse import save_npz,load_npz,issparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
from gensim.models import Word2Vec,FastText
from xgboost import XGBClassifier
from executar_divisoes_20261105 import tokenizar,hash_estavel,vetor_documento,PARAMS,sha,dump,fingerprint
from protocolo_robustez_20261106 import load_data,TG,R
E=TG/'experimentos_divisoes_2026_11_05';N=TG/'revisao_desenvolvimento_2026_11_03/notebooks'
BASES=['fakebr','boatosbr','faketrue','pteu'];METHODS=['tfidf','word2vec','fasttext','bertimbau','e5','qwen3'];SEEDS=[42,43,44,45,46]
WORKER_LABEL=os.environ.get('TG_WORKER','')
WORKER=bool(WORKER_LABEL)
SELECTED_BASES=BASES[1:] if WORKER_LABEL=='outras_bases' else ['fakebr'] if WORKER_LABEL in ['fakebr_46','fakebr_45','fakebr_sem45'] else BASES
SELECTED_SEEDS=[46] if WORKER_LABEL=='fakebr_46' else [45] if WORKER_LABEL=='fakebr_45' else [42,43,44,46] if WORKER_LABEL=='fakebr_sem45' else SEEDS
EXPECTED_FITS=len(SELECTED_BASES)*len(SELECTED_SEEDS)*6
def status_path():return R/'evidencias'/('ANDAMENTO_'+WORKER_LABEL.upper()+'.json' if WORKER else 'ANDAMENTO.json')
def csvpath(name):return R/('parciais_'+WORKER_LABEL+'_'+name if WORKER else name)
def store_matrix(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    if issparse(a):p=p.with_suffix('.npz');save_npz(p,a)
    else:p=p.with_suffix('.npy');np.save(p,a)
    return {'path':str(p.relative_to(R)),'sha256':sha(p),'shape':list(a.shape),'sparse':issparse(a)}
def load_matrix(rec):
    p=R/rec['path'];assert sha(p)==rec['sha256'];return load_npz(p) if rec['sparse'] else np.load(p,allow_pickle=False)
def traditional(d,tr,te,method,targets):
    train=d.iloc[tr].text.tolist();test=d.iloc[te].text.tolist()
    if method=='tfidf':
        v=TfidfVectorizer(tokenizer=tokenizar,token_pattern=None,preprocessor=None,lowercase=False,max_features=5000,ngram_range=(1,2),max_df=.85,min_df=2,dtype=np.float32)
        a=v.fit_transform(train);b=v.transform(test);others={k:v.transform(x.text.tolist()) for k,x in targets.items()}
        return a,b,others,{'vocabulary_size':len(v.vocabulary_),'learned_from_train_only':True}
    pt=[tokenizar(t) for t in train];pe=[tokenizar(t) for t in test]
    args=dict(sentences=pt,vector_size=100,window=5,min_count=1,workers=1,sg=1,seed=42,epochs=10,negative=5,hs=0,sample=.001,ns_exponent=.75,alpha=.025,min_alpha=.0001,hashfxn=hash_estavel)
    model=FastText(**args,min_n=3,max_n=6,bucket=200000) if method=='fasttext' else Word2Vec(**args)
    def transform(words):return np.asarray([vetor_documento(p,model,method=='fasttext') for p in words],dtype=np.float32)
    a=transform(pt);b=transform(pe);others={k:transform([tokenizar(t) for t in x.text]) for k,x in targets.items()};coverage={}
    for split,words,arr in [('train',pt,a),('test',pe,b)]:
        total=sum(map(len,words));known=sum(w in model.wv.key_to_index for doc in words for w in doc);coverage[split]={'tokens':total,'known_tokens':known,'known_fraction':known/max(total,1),'zero_vectors':int(np.sum(~np.any(arr,axis=1)))}
    del model;gc.collect();return a,b,others,{'vocabulary_coverage':coverage,'learned_from_train_only':True}
def note(message):
    p=status_path();status=json.loads(p.read_text(encoding='utf8')) if p.exists() else {};status.update(last_event=message,execution='CPU local, encoders congelados originais, sem extração nova');dump(p,status)
    if WORKER:return
    head=(R/'CONTINUIDADE_INICIAL.md').read_text(encoding='utf8')
    text=head+'\n\n## Último marco da execução\n'+message+'\nDetalhes e contadores em evidencias/ANDAMENTO.json. Retomar pelo executor, que confere assinatura e hashes antes de aproveitar saídas concluídas. Não publicar valores parciais como resultado final.\n'
    for q in [TG/'CONTINUIDADE_TG.md',R/'CONTINUIDADE_ROBUSTEZ.md']:q.write_text(text,encoding='utf8')
def evaluate(d,tr,te,model,Xte,out,method,cfg,extra,matrix_rec):
    prob=model.predict_proba(Xte)[:,1];pred=(prob>=.5).astype(int);df=d.iloc[te][['doc_id','group_id','canonical_sha256','label']].copy();df['prob_fake']=prob;df['prediction']=pred
    pp=out/(method+'_predicoes.csv');df.to_csv(pp,index=False);model.save_model(out/(method+'_xgboost.json'))
    result={**cfg,**extra,'method':method,'n_train':len(tr),'n_test':len(te),'macro_f1':float(f1_score(df.label,pred,average='macro')),'accuracy':float(accuracy_score(df.label,pred)),'classification_report':classification_report(df.label,pred,labels=[0,1],target_names=['Verdadeira','Falsa'],output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(df.label,pred,labels=[0,1]).tolist(),'predictions_sha256':sha(pp),'matrices':matrix_rec}
    dump(out/(method+'_resultado.json'),result);return result
def run():
    all_data={b:load_data(b) for b in BASES}
    original_fingerprints={b:fingerprint(pd.read_csv(R/'dados'/(b+'.csv'),keep_default_na=False,dtype={'doc_id':str,'pair_id':str,'group_id':str})) for b in BASES}
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','gensim','xgboost','datasketch','psutil']};versions.update(python=platform.python_version(),platform=platform.platform());dump(R/'evidencias/VERSOES.json',versions)
    audit=json.loads((E/'evidencias/VALIDACAO_CACHES_COMPLETA.json').read_text(encoding='utf8'));caches={}
    for base,d in all_data.items():
        assert sha(R/'dados'/(base+'.csv'))==sha(E/'dados'/(base+'.csv'))
        caches[base]={}
        for method in METHODS[3:]:
            p=E/'vetores_contextuais'/base/method;record=next(r for r in audit if r['base']==base and r['method']==method);spec=json.loads((p/'spec.json').read_text(encoding='utf8'));assert spec['fingerprint']==original_fingerprints[base] and sha(p/'all.npy')==record['all_sha256']
            arr=np.load(p/'all.npy',mmap_mode='r',allow_pickle=False);assert np.isfinite(arr).all();caches[base][method]=arr
    dump(R/'evidencias/VALIDACAO_CACHES_ENTRADA.json',{'status':'PASS','arrays':12,'source':'experimentos_divisoes_2026_11_05','new_extraction':False})
    rows=[];length_rows=[];completed=0
    for base,d in all_data.items():
        if base not in SELECTED_BASES:continue
        for seed in SELECTED_SEEDS:
            tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(d,groups=d.group_id));train=d.iloc[tr];test=d.iloc[te]
            assert not set(train.group_id)&set(test.group_id) and not set(train.canonical_sha256)&set(test.canonical_sha256)
            assert train.label.nunique()==test.label.nunique()==2
            out=R/'resultados/repeticoes'/base/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
            cfg={'base':base,'split_seed':seed,'train_seed':42,'test_fraction_groups':.2,'source_sha256':sha(R/'dados'/(base+'.csv')),'group_mapping_sha256':sha(R/'dados/GRUPOS_REFORCADOS_FAKEBR.csv') if base=='fakebr' else 'original','params_xgboost':PARAMS,'train_ids':train.doc_id.tolist(),'test_ids':test.doc_id.tolist()}
            cfgsig=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest();dump(out/'CONFIGURACAO.json',cfg)
            part=d[['doc_id','group_id','canonical_sha256','label']].copy();part['split']='train';part.loc[te,'split']='test';part.to_csv(out/'particoes.csv',index=False)
            for method in METHODS:
                rp=out/(method+'_resultado.json')
                if rp.exists():
                    result=json.loads(rp.read_text(encoding='utf8'));assert result['configuration_signature']==cfgsig and result['predictions_sha256']==sha(out/(method+'_predicoes.csv'))
                    print('RETOMADO',base,seed,method,flush=True)
                else:
                    print('INICIANDO',base,seed,method,flush=True);note(f'Início: {base}, seed{seed}, {method}. Concluídos {completed}/120 ajustes das seis representações.')
                    t=time.perf_counter();targets={k:all_data[k] for k in BASES[:3] if k!=base} if seed==42 and base in BASES[:3] else {}
                    if method in METHODS[:3]:Xtr,Xte,other,extra=traditional(d,tr,te,method,targets)
                    else:Xtr,Xte,other,extra=caches[base][method][tr],caches[base][method][te],{}, {'frozen_original_embeddings':True}
                    extract_time=time.perf_counter()-t;matrices={}
                    if method in METHODS[:3]:
                        matrices={'train':store_matrix(out/method/'train',Xtr),'test':store_matrix(out/method/'test',Xte)}
                        for target,arr in other.items():matrices['target_'+target]=store_matrix(out/method/('target_'+target),arr)
                    model=XGBClassifier(**PARAMS);t=time.perf_counter();model.fit(Xtr,train.label.to_numpy());fit_time=time.perf_counter()-t
                    result=evaluate(d,tr,te,model,Xte,out,method,{'configuration_signature':cfgsig},dict(extra,representation_seconds_local=extract_time,fit_seconds_local=fit_time),matrices)
                    if seed==42 and base!='fakebr':
                        old=pd.read_csv(N/'referencias_resultados'/base/(method+'.csv'),dtype={'doc_id':str});now=pd.read_csv(out/(method+'_predicoes.csv'),dtype={'doc_id':str});assert old.doc_id.tolist()==now.doc_id.tolist()
                        result['historical_seed42_disagreements']=int(sum(old.prediction.to_numpy()!=now.prediction.to_numpy()));dump(rp,result)
                    del Xtr,Xte,other,model;gc.collect()
                rows.append({'base':base,'split_seed':seed,'method':method,**{k:result[k] for k in ['macro_f1','accuracy','n_train','n_test','representation_seconds_local','fit_seconds_local']}});completed+=1
                pd.DataFrame(rows).to_csv(csvpath('REPETICOES_80_20.csv'),index=False)
                dump(status_path(),{'status':'EXECUTANDO_REPETICOES','completed_representation_fits':completed,'planned_representation_fits':EXPECTED_FITS,'split_seeds':SELECTED_SEEDS,'train_seed':42})
                note(f'Concluído: {base}, seed{seed}, {method}; macro-F1={result["macro_f1"]:.4f}; {completed}/120 ajustes.')
                print('CONCLUIDO',base,seed,method,round(result['macro_f1'],4),f'{completed}/120',flush=True)
            # Diagnóstico com uma única característica e a mesma configuração XGBoost.
            lp=out/'comprimento_resultado.json'
            if not lp.exists():
                vals=np.log1p(d.text.map(lambda t:len(t.split())).to_numpy(dtype=np.float32)).reshape(-1,1);model=XGBClassifier(**PARAMS);t=time.perf_counter();model.fit(vals[tr],train.label.to_numpy());secs=time.perf_counter()-t
                evaluate(d,tr,te,model,vals[te],out,'comprimento',{'configuration_signature':cfgsig},{'representation_seconds_local':0.,'fit_seconds_local':secs,'single_feature':'log1p(words_by_spaces)'},{'test':store_matrix(out/'comprimento/test',vals[te])});del model
            lr=json.loads(lp.read_text(encoding='utf8'));major=int(train.label.value_counts().idxmax());pred=np.full(len(test),major)
            length_rows.append({'base':base,'split_seed':seed,'length_macro_f1':lr['macro_f1'],'length_accuracy':lr['accuracy'],'majority_macro_f1':float(f1_score(test.label,pred,average='macro')),'majority_accuracy':float(accuracy_score(test.label,pred))});pd.DataFrame(length_rows).to_csv(csvpath('CONTROLE_COMPRIMENTO.csv'),index=False)
    assert len(rows)==EXPECTED_FITS and len(length_rows)==EXPECTED_FITS//6
    df=pd.DataFrame(rows);summary=df.groupby(['base','method']).agg(mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),min_macro_f1=('macro_f1','min'),max_macro_f1=('macro_f1','max'),repetitions=('macro_f1','size')).reset_index();summary.to_csv(csvpath('RESUMO_REPETICOES.csv'),index=False)
    dump(status_path(),{'status':'REPETICOES_CONCLUIDAS','completed_representation_fits':len(rows),'length_fits':len(length_rows),'historical_reference_not_added_to_count':True,'new_encoder_extractions':0,'next':'concluir auditoria cruzada e executar controles de amostras/fontes/transferência'})
    note('Concluídas cinco repetições nas quatro bases:120 ajustes e20 diagnósticos de comprimento. Falta auditoria final, controles selecionados, transferência e artigo privado.')
    print('FINALIZADO120+20',flush=True)
if __name__=='__main__':
    try:run()
    except Exception as e:
        dump(R/'evidencias'/('FALHA_WORKER.json' if WORKER else 'FALHA_EXECUCAO.json'),{'error':str(e),'traceback':traceback.format_exc(),'resume':'Executar novamente; saídas validadas serão aproveitadas.'});note('Falha registrada em evidencias; preservar saídas e corrigir antes de retomar.');raise
