"""Auditoria independente de métricas, partições e recuperação dos modelos."""
from pathlib import Path
import sys,json,hashlib,gc,itertools
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'testdeps'))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
from xgboost import XGBClassifier
from executar_robustez_20261106 import TG,R,E,N,BASES,METHODS,PARAMS,load_matrix
from executar_divisoes_20261105 import dump,sha,intervalos,tokenizar
from protocolo_robustez_20261106 import load_data
def check(d,tr,te,out,m,record,X):
    p=pd.read_csv(out/(m+'_predicoes.csv'),keep_default_na=False,dtype={'doc_id':str,'group_id':str});test=d.iloc[te]
    assert p.doc_id.tolist()==test.doc_id.tolist() and p.label.tolist()==test.label.tolist() and p.group_id.tolist()==test.group_id.tolist()
    assert sha(out/(m+'_predicoes.csv'))==record['predictions_sha256']
    model=XGBClassifier(n_jobs=2);model.load_model(out/(m+'_xgboost.json'));assert model.classes_.tolist()==[0,1] and model.n_features_in_==X.shape[1]
    prob=model.predict_proba(X)[:,1];pred=(prob>=.5).astype(int);err=float(np.max(np.abs(prob-p.prob_fake.to_numpy())))
    assert err<1e-6 and np.array_equal(pred,p.prediction.to_numpy())
    cr=classification_report(test.label,pred,labels=[0,1],target_names=['Verdadeira','Falsa'],output_dict=True,zero_division=0);cm=confusion_matrix(test.label,pred,labels=[0,1]).tolist()
    assert abs(f1_score(test.label,pred,average='macro')-record['macro_f1'])<1e-12 and abs(accuracy_score(test.label,pred)-record['accuracy'])<1e-12
    assert cm==record['confusion_matrix']
    for label in ['Verdadeira','Falsa']:
        for metric in ['precision','recall','f1-score','support']:assert abs(cr[label][metric]-record['classification_report'][label][metric])<1e-12
    del model;gc.collect();return p,cr,{'predictions_reproduced':True,'max_probability_difference':err}
def feature_matrix(d,base,method,te,record):
    if method in METHODS[:3]+['comprimento']:return load_matrix(record['matrices']['test'])
    return np.load(E/'vetores_contextuais'/base/method/'all.npy',mmap_mode='r',allow_pickle=False)[te]
def run():
    data={b:load_data(b) for b in BASES};rows=[];counts=[];replays=[];classes=[];confusions=[];length=[];source_rows=[]
    assert not json.loads((R/'evidencias/AUDITORIA_GRUPOS_INTERNOS.json').read_text(encoding='utf8'))['need_review_before_publication']
    assert json.loads((R/'evidencias/AUDITORIA_GRUPOS_PT_EU.json').read_text(encoding='utf8'))['status']=='PASS'
    near=pd.read_csv(R/'evidencias/DUPLICATAS_PROXIMAS_INTERNAS.csv',dtype={'id_a':str,'id_b':str});ptnear=pd.read_csv(R/'evidencias/DUPLICATAS_PROXIMAS_PT_EU.csv',dtype={'id_a':str,'id_b':str});ptnear['base']='pteu';near=pd.concat([near,ptnear],ignore_index=True)
    for b,d in data.items():
        assert sha(R/'dados'/(b+'.csv'))==sha(E/'dados'/(b+'.csv'))
        raw=pd.read_csv(R/'dados'/(b+'.csv'),keep_default_na=False,dtype={'doc_id':str,'group_id':str});assert d.doc_id.tolist()==raw.doc_id.tolist() and d.text.tolist()==raw.text.tolist() and d.label.tolist()==raw.label.tolist()
        byid=d.set_index('doc_id')
        for _,p in near[near.base==b].iterrows():assert byid.loc[p.id_a].group_id==byid.loc[p.id_b].group_id
        meta=pd.read_csv(R/'dados'/(b+'_metadados.csv'),keep_default_na=False,dtype={'doc_id':str,'group_id':str});assert meta.doc_id.tolist()==d.doc_id.tolist() and meta.group_id.tolist()==d.group_id.tolist()
        for seed in [42,43,44,45,46]:
            out=R/'resultados/repeticoes'/b/f'seed_{seed}';cfg=json.loads((out/'CONFIGURACAO.json').read_text(encoding='utf8'));sig=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest();assert cfg['params_xgboost']==PARAMS
            part=pd.read_csv(out/'particoes.csv',dtype={'doc_id':str,'group_id':str});assert part.doc_id.tolist()==d.doc_id.tolist() and part.group_id.tolist()==d.group_id.tolist();tr=np.flatnonzero(part.split=='train');te=np.flatnonzero(part.split=='test')
            assert cfg['train_ids']==d.iloc[tr].doc_id.tolist() and cfg['test_ids']==d.iloc[te].doc_id.tolist()
            assert not set(d.iloc[tr].group_id)&set(d.iloc[te].group_id) and not set(d.iloc[tr].canonical_sha256)&set(d.iloc[te].canonical_sha256)
            assert d.iloc[tr].label.nunique()==d.iloc[te].label.nunique()==2
            counts.append({'base':b,'split_seed':seed,'train_groups':d.iloc[tr].group_id.nunique(),'test_groups':d.iloc[te].group_id.nunique(),'train_docs':len(tr),'test_docs':len(te),'train_true':int(sum(d.iloc[tr].label==0)),'train_false':int(sum(d.iloc[tr].label==1)),'test_true':int(sum(d.iloc[te].label==0)),'test_false':int(sum(d.iloc[te].label==1)),'groups_hashes_disjoint':True})
            probs={}
            for m in METHODS:
                rec=json.loads((out/(m+'_resultado.json')).read_text(encoding='utf8'));assert rec['configuration_signature']==sig and rec['n_train']==len(tr) and rec['n_test']==len(te)
                X=feature_matrix(d,b,m,te,rec);p,cr,proof=check(d,tr,te,out,m,rec,X);probs[m]=p.prob_fake.to_numpy();replays.append({'scope':'primary','base':b,'seed':seed,'method':m,**proof});rows.append({'base':b,'split_seed':seed,'method':m,**{k:rec[k] for k in ['macro_f1','accuracy','n_train','n_test','representation_seconds_local','fit_seconds_local']}})
                if seed==42 and b!='fakebr':
                    old=pd.read_csv(N/'referencias_resultados'/b/(m+'.csv'),dtype={'doc_id':str});assert old.doc_id.tolist()==p.doc_id.tolist() and old.prediction.tolist()==p.prediction.tolist()
                for cls,label in enumerate(['Verdadeira','Falsa']):classes.append({'scope':'primary','base':b,'seed':seed,'method':m,'class':cls,**cr[label]})
                confusions.append({'scope':'primary','base':b,'seed':seed,'method':m,'tn':rec['confusion_matrix'][0][0],'fp':rec['confusion_matrix'][0][1],'fn':rec['confusion_matrix'][1][0],'tp':rec['confusion_matrix'][1][1]});del X
            boot=intervalos(d.iloc[te],probs);boot['conditioning']='Grupos de notícias e fontes do teste fixas; não é intervalo da média dos cinco sorteios.';dump(out/'bootstrap_grupos.json',boot)
            rec=json.loads((out/'comprimento_resultado.json').read_text(encoding='utf8'));X=feature_matrix(d,b,'comprimento',te,rec);p,cr,proof=check(d,tr,te,out,'comprimento',rec,X);replays.append({'scope':'length','base':b,'seed':seed,**proof});major=int(d.iloc[tr].label.value_counts().idxmax());yp=np.full(len(te),major);length.append({'base':b,'split_seed':seed,'length_macro_f1':rec['macro_f1'],'length_accuracy':rec['accuracy'],'majority_macro_f1':float(f1_score(d.iloc[te].label,yp,average='macro')),'majority_accuracy':float(accuracy_score(d.iloc[te].label,yp))})
            for cls,label in enumerate(['Verdadeira','Falsa']):classes.append({'scope':'length','base':b,'seed':seed,'method':'comprimento','class':cls,**cr[label]})
            confusions.append({'scope':'length','base':b,'seed':seed,'method':'comprimento','tn':rec['confusion_matrix'][0][0],'fp':rec['confusion_matrix'][0][1],'fn':rec['confusion_matrix'][1][0],'tp':rec['confusion_matrix'][1][1]})
            sr=pd.read_csv(R/'resultados/diagnostico_fonte'/b/f'seed_{seed}_predicoes.csv',dtype={'doc_id':str,'group_id':str});assert sr.doc_id.tolist()==d.iloc[te].doc_id.tolist() and sr.group_id.tolist()==d.iloc[te].group_id.tolist()
            mapping={site:int(f.label.value_counts().idxmax()) for site,f in meta.iloc[tr].groupby('source')};sp=meta.iloc[te].source.map(mapping).fillna(major).to_numpy(dtype=int);assert np.array_equal(sp,sr.prediction)
            source_rows.append({'base':b,'split_seed':seed,'macro_f1':float(f1_score(d.iloc[te].label,sp,average='macro')),'accuracy':float(accuracy_score(d.iloc[te].label,sp)),'n_train':len(tr),'n_test':len(te),'unseen_source_docs':int(sum(~meta.iloc[te].source.isin(mapping))),'fitted_on_train_only':True})
            print('AUDITADO_PRINCIPAL',b,seed,flush=True)
    assert len(rows)==120 and len(replays)==140 and len(counts)==20
    allrows=pd.DataFrame(rows);assert all(allrows.groupby(['base','method']).size()==5)
    summary=allrows.groupby(['base','method']).agg(mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),min_macro_f1=('macro_f1','min'),max_macro_f1=('macro_f1','max'),repetitions=('macro_f1','size')).reset_index()
    # Os CSVs finais são reconstruídos das predições e registros conferidos.
    for name,frame in [('REPETICOES_80_20.csv',allrows),('RESUMO_REPETICOES.csv',summary),('DIVISOES_80_20.csv',pd.DataFrame(counts)),('CONTROLE_COMPRIMENTO.csv',pd.DataFrame(length)),('DIAGNOSTICO_FONTES.csv',pd.DataFrame(source_rows))]:frame.to_csv(R/name,index=False)
    controls=[]
    for kind in ['comprimento_pareado','fontes_separadas']:
        for b,d in data.items():
            out=R/'resultados'/kind/b;sel=json.loads((out/('SELECAO.json' if kind=='comprimento_pareado' else 'VIABILIDADE.json')).read_text(encoding='utf8'))
            if not sel['eligible']:continue
            part=pd.read_csv(out/'particoes.csv',keep_default_na=False,dtype={'doc_id':str,'group_id':str});assert part.doc_id.tolist()==d.doc_id.tolist() and part.group_id.tolist()==d.group_id.tolist();tr=np.flatnonzero(part.split=='train');te=np.flatnonzero(part.split=='test')
            assert not set(d.iloc[tr].group_id)&set(d.iloc[te].group_id)
            if kind=='fontes_separadas':assert not set(part.iloc[tr].source)&set(part.iloc[te].source)
            else:
                assert set(tr)<=set(np.flatnonzero(d.split=='train')) and set(te)<=set(np.flatnonzero(d.split=='test'))
                for split in ['train','test']:
                    for _,f in part[part.split==split].groupby('bin'):assert int(sum(f.label==0))==int(sum(f.label==1))
            cfg=json.loads((out/'CONFIGURACAO_EXECUCAO.json').read_text(encoding='utf8'));sig=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest();assert cfg['partition_sha256']==sha(out/'particoes.csv')
            major=int(d.iloc[tr].label.value_counts().idxmax());dummy=float(f1_score(d.iloc[te].label,np.full(len(te),major),average='macro'))
            probs={};baselines={}
            for m in METHODS+['comprimento']:
                rec=json.loads((out/(m+'_resultado.json')).read_text(encoding='utf8'));assert rec['configuration_signature']==sig
                X=feature_matrix(d,b,m,te,rec);p,cr,proof=check(d,tr,te,out,m,rec,X);replays.append({'scope':kind,'base':b,'method':m,**proof});row={'condition':kind,'base':b,'method':m,'macro_f1':rec['macro_f1'],'accuracy':rec['accuracy'],'n_train':len(tr),'n_test':len(te),'test_true':int(sum(d.iloc[te].label==0)),'test_false':int(sum(d.iloc[te].label==1)),'majority_train_macro_f1':dummy}
                if m in METHODS:
                    probs[m]=p.prob_fake.to_numpy()
                    if kind=='comprimento_pareado':
                        ref=pd.read_csv(R/'resultados/repeticoes'/b/'seed_42'/(m+'_predicoes.csv'),dtype={'doc_id':str}).set_index('doc_id').loc[d.iloc[te].doc_id];baselines[m]=ref.prediction.to_numpy();row['full_train_same_selected_test_macro_f1']=float(f1_score(ref.label,ref.prediction,average='macro'));row['delta_vs_full_train_same_test']=row['macro_f1']-row['full_train_same_selected_test_macro_f1']
                for cls,label in enumerate(['Verdadeira','Falsa']):classes.append({'scope':kind,'base':b,'seed':42 if kind=='comprimento_pareado' else sel['split_seed'],'method':m,'class':cls,**cr[label]})
                confusions.append({'scope':kind,'base':b,'seed':42 if kind=='comprimento_pareado' else sel['split_seed'],'method':m,'tn':rec['confusion_matrix'][0][0],'fp':rec['confusion_matrix'][0][1],'fn':rec['confusion_matrix'][1][0],'tp':rec['confusion_matrix'][1][1]})
                controls.append(row);del X
            boot=intervalos(d.iloc[te],probs,baselines if len(baselines)==6 else None)
            for x in boot['diferencas_pareadas_exploratorias']:x['contrast']='treino pareado menos treino completo, mesmo teste selecionado, seed42 com grupos atuais'
            boot['conditioning']='Grupos noticiários e fontes usadas fixas; não mede incerteza entre novos sites.';dump(out/'bootstrap_teste.json',boot)
            print('AUDITADO_CONTROLE',kind,b,flush=True)
    assert len(controls)==35;pd.DataFrame(controls).to_csv(R/'RESULTADOS_CONTROLES.csv',index=False)
    transfer=[];targetvocab={}
    for src,tgt in itertools.permutations(BASES[:3],2):
        external_probs={};internal_predictions={}
        out=R/'resultados/transferencia'/f'{src}_para_{tgt}';part=pd.read_csv(out/'ALVO_CONGELADO.csv',dtype={'doc_id':str,'group_id':str});d=data[tgt];assert part.doc_id.tolist()==d.doc_id.tolist() and part.group_id.tolist()==d.group_id.tolist() and part.evaluate.dtype==bool;te=np.flatnonzero(part.evaluate);assert set(te)<=set(np.flatnonzero(d.split=='test'))
        if src not in targetvocab:targetvocab[src]=set(w for t in data[src][data[src].split=='train'].text for w in tokenizar(t))
        tokens=[tokenizar(t) for t in d.iloc[te].text];nt=sum(map(len,tokens));known=sum(w in targetvocab[src] for words in tokens for w in words)
        for m in METHODS:
            origin=R/'resultados/repeticoes'/src/'seed_42';fit=json.loads((origin/(m+'_resultado.json')).read_text(encoding='utf8'));rec=json.loads((out/(m+'_resultado.json')).read_text(encoding='utf8'));assert rec['source_model_sha256']==sha(origin/(m+'_xgboost.json'))==sha(out/(m+'_modelo_fonte.json'))
            X=load_matrix(fit['matrices']['target_'+tgt])[te] if m in METHODS[:3] else np.load(E/'vetores_contextuais'/tgt/m/'all.npy',mmap_mode='r',allow_pickle=False)[te]
            model=XGBClassifier(n_jobs=2);model.load_model(out/(m+'_modelo_fonte.json'));prob=model.predict_proba(X)[:,1];p=pd.read_csv(out/(m+'_predicoes.csv'),dtype={'doc_id':str,'group_id':str});assert p.doc_id.tolist()==d.iloc[te].doc_id.tolist() and p.label.tolist()==d.iloc[te].label.tolist() and p.group_id.tolist()==d.iloc[te].group_id.tolist();assert np.array_equal((prob>=.5).astype(int),p.prediction) and np.max(np.abs(prob-p.prob_fake.to_numpy()))<1e-6 and sha(out/(m+'_predicoes.csv'))==rec['predictions_sha256']
            assert abs(f1_score(p.label,p.prediction,average='macro')-rec['macro_f1'])<1e-12;ref=pd.read_csv(R/'resultados/repeticoes'/tgt/'seed_42'/(m+'_predicoes.csv'),dtype={'doc_id':str}).set_index('doc_id').loc[p.doc_id];internal=float(f1_score(ref.label,ref.prediction,average='macro'));assert abs(internal-rec['internal_target_macro_f1_same_test'])<1e-12
            external_probs[m]=prob.copy();internal_predictions[m]=ref.prediction.to_numpy()
            extra={}
            if m in ['word2vec','fasttext']:extra={'target_tokens':nt,'source_known_tokens':known,'known_fraction':known/max(nt,1),'oov_vectors_synthesized_by_subwords':m=='fasttext','zero_target_document_vectors':int(sum(~np.any(X,axis=1)))}
            elif m=='tfidf':extra={'zero_target_document_vectors':int(sum(X.getnnz(axis=1)==0))}
            if extra:dump(out/(m+'_cobertura_alvo.json'),extra)
            cr=classification_report(p.label,p.prediction,labels=[0,1],target_names=['Verdadeira','Falsa'],output_dict=True,zero_division=0);cm=confusion_matrix(p.label,p.prediction,labels=[0,1]).tolist();rec['classification_report']=cr;rec['confusion_matrix']=cm;dump(out/(m+'_resultado.json'),rec)
            for cls,label in enumerate(['Verdadeira','Falsa']):classes.append({'scope':'transfer','base':tgt,'source':src,'seed':42,'method':m,'class':cls,**cr[label]})
            confusions.append({'scope':'transfer','base':tgt,'source':src,'seed':42,'method':m,'tn':cm[0][0],'fp':cm[0][1],'fn':cm[1][0],'tp':cm[1][1]})
            transfer.append(rec);replays.append({'scope':'transfer','source':src,'target':tgt,'method':m,'predictions_reproduced':True});del X,model;gc.collect()
        recovered_boot=intervalos(d.iloc[te],external_probs,internal_predictions);saved_boot=json.loads((out/'bootstrap_teste.json').read_text(encoding='utf8'))
        assert recovered_boot['intervalos']==saved_boot['intervalos']
        for current,previous in zip(recovered_boot['diferencas_pareadas_exploratorias'],saved_boot['diferencas_pareadas_exploratorias']):
            assert current['method']==previous['method'] and abs(current['lower']-previous['lower'])<1e-12 and abs(current['upper']-previous['upper'])<1e-12
        print('AUDITADA_TRANSFERENCIA',src,tgt,flush=True)
    assert len(transfer)==36;pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in transfer]).to_csv(R/'RESULTADOS_TRANSFERENCIA.csv',index=False);pd.DataFrame(classes).to_csv(R/'METRICAS_POR_CLASSE.csv',index=False);pd.DataFrame(confusions).to_csv(R/'MATRIZES_CONFUSAO_ROBUSTEZ.csv',index=False);pd.DataFrame([row for row in confusions if row['scope']=='primary']).to_csv(R/'MATRIZES_CONFUSAO_PRINCIPAIS.csv',index=False)
    # Conferência do bootstrap por cálculo direto em uma condição, sem usar contagens agregadas.
    b='faketrue';d=data[b];out=R/'resultados/repeticoes'/b/'seed_42';p=pd.read_csv(out/'tfidf_predicoes.csv',dtype={'group_id':str});groups=sorted(p.group_id.unique());rng=np.random.default_rng(42);vals=[]
    for _ in range(2000):
        sample=rng.integers(len(groups),size=len(groups));idx=np.concatenate([np.flatnonzero(p.group_id.to_numpy()==groups[i]) for i in sample]);y=p.label.to_numpy()[idx]
        if len(np.unique(y))==2:vals.append(f1_score(y,p.prediction.to_numpy()[idx],average='macro',labels=[0,1],zero_division=0))
    expected=json.loads((out/'bootstrap_grupos.json').read_text(encoding='utf8'))['intervalos']['tfidf'];assert abs(float(np.quantile(vals,.025))-expected['lower'])<1e-12 and abs(float(np.quantile(vals,.975))-expected['upper'])<1e-12
    notebook=json.loads((R/'evidencias/VALIDACAO_NOTEBOOK.json').read_text(encoding='utf8'));assert notebook['status']=='PASS' and notebook['actual_additional_validation_fits']==6
    v={'status':'PASS','primary_fits':120,'primary_conditions':20,'means_sd':24,'repetitions_each':5,'length_primary_fits':20,'extra_representation_fits':30,'extra_length_fits':5,'scientific_xgboost_fits':175,'source_majority_evaluations':20,'transfer_evaluations':36,'transfer_new_fits':0,'notebook_validation_fits_separate':6,'models_reloaded':175,'transfer_model_replays':36,'all_predictions_and_metrics':'PASS','bootstrap_direct_check':{'base':'faketrue','seed':42,'method':'tfidf','resamples':2000,'result':'PASS'},'all_group_hash_checks':'PASS','near_duplicates_in_distinct_groups_after_fix':0,'fakebr_groups':3587,'fakebr_documents_labels_unchanged':True,'historical_seed42_predictions_reproduced_other_datasets':18,'new_encoder_extractions':0,'replicates_overlap':'Descriptive mean and sample SD; not independent observations or CI of mean.'}
    dump(R/'evidencias/REPRODUCAO_MODELOS.json',replays);dump(R/'evidencias/CONFERENCIA_FINAL.json',v);print(json.dumps(v,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':run()
