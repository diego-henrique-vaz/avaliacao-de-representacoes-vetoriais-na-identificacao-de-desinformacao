from pathlib import Path
import sys,json,time,gc,hashlib,traceback
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'testdeps'))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score,accuracy_score
from xgboost import XGBClassifier
from executar_robustez_20261106 import TG,R,E,N,BASES,METHODS,PARAMS,traditional,store_matrix,load_matrix,evaluate
from executar_divisoes_20261105 import dump,sha,intervalos
from protocolo_robustez_20261106 import load_data
def run_controls():
    data={b:load_data(b) for b in BASES};rows=[];diagnostics=[];done=0
    for kind in ['comprimento_pareado','fontes_separadas']:
        for b,d in data.items():
            out=R/'resultados'/kind/b;selection=json.loads((out/('SELECAO.json' if kind=='comprimento_pareado' else 'VIABILIDADE.json')).read_text(encoding='utf8'))
            if not selection['eligible']:continue
            part=pd.read_csv(out/'particoes.csv',keep_default_na=False,dtype={'doc_id':str,'group_id':str});assert part.doc_id.tolist()==d.doc_id.tolist()
            tr=np.flatnonzero(part.split.eq('train'));te=np.flatnonzero(part.split.eq('test'));assert d.iloc[tr].label.nunique()==d.iloc[te].label.nunique()==2
            assert not set(d.iloc[tr].group_id)&set(d.iloc[te].group_id)
            if kind=='fontes_separadas':assert not set(part.iloc[tr].source)&set(part.iloc[te].source)
            cfg={'condition':kind,'base':b,'data_sha256':sha(R/'dados'/(b+'.csv')),'partition_sha256':sha(out/'particoes.csv'),'params_xgboost':PARAMS};sig=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest();dump(out/'CONFIGURACAO_EXECUCAO.json',cfg)
            probs={};baseline={}
            for method in METHODS+['comprimento']:
                rp=out/(method+'_resultado.json')
                if rp.exists():
                    result=json.loads(rp.read_text(encoding='utf8'));assert result['configuration_signature']==sig and sha(out/(method+'_predicoes.csv'))==result['predictions_sha256']
                    saved=pd.read_csv(out/(method+'_predicoes.csv'),dtype={'doc_id':str});assert saved.doc_id.tolist()==d.iloc[te].doc_id.tolist()
                else:
                    print('INICIANDO_CONTROLE',kind,b,method,flush=True);t=time.perf_counter()
                    if method in METHODS[:3]:a,z,_,extra=traditional(d,tr,te,method,{})
                    elif method=='comprimento':
                        x=np.log1p(d.text.map(lambda t:len(t.split())).to_numpy(dtype=np.float32)).reshape(-1,1);a,z,extra=x[tr],x[te],{'single_feature':'log1p(words_by_spaces)'}
                    else:
                        x=np.load(E/'vetores_contextuais'/b/method/'all.npy',mmap_mode='r',allow_pickle=False);a,z,extra=x[tr],x[te],{'frozen_original_embeddings':True}
                    secs=time.perf_counter()-t;matrices={'train':store_matrix(out/method/'train',a),'test':store_matrix(out/method/'test',z)} if method in METHODS[:3]+['comprimento'] else {}
                    model=XGBClassifier(**PARAMS);t=time.perf_counter();model.fit(a,d.iloc[tr].label.to_numpy());fit=time.perf_counter()-t
                    result=evaluate(d,tr,te,model,z,out,method,{'configuration_signature':sig},dict(extra,representation_seconds_local=secs,fit_seconds_local=fit),matrices)
                    saved=pd.read_csv(out/(method+'_predicoes.csv'),dtype={'doc_id':str});del a,z,model;gc.collect()
                probs[method]=saved.prob_fake.to_numpy();row={'condition':kind,'base':b,'method':method,'macro_f1':result['macro_f1'],'accuracy':result['accuracy'],'n_train':len(tr),'n_test':len(te),'test_true':int(sum(d.iloc[te].label==0)),'test_false':int(sum(d.iloc[te].label==1))}
                if kind=='comprimento_pareado' and method in METHODS:
                    reference=R/'resultados/repeticoes'/b/'seed_42'/(method+'_predicoes.csv') if b=='fakebr' else N/'referencias_resultados'/b/(method+'.csv')
                    if reference.exists():
                        old=pd.read_csv(reference,dtype={'doc_id':str}).set_index('doc_id');ref=old.loc[d.iloc[te].doc_id];assert ref.label.tolist()==d.iloc[te].label.tolist();baseline[method]=ref.prediction.to_numpy();row['full_train_same_selected_test_macro_f1']=float(f1_score(ref.label,ref.prediction,average='macro'));row['delta_vs_full_train_same_test']=row['macro_f1']-row['full_train_same_selected_test_macro_f1']
                rows.append(row);done+=1;pd.DataFrame(rows).to_csv(R/'RESULTADOS_CONTROLES.csv',index=False);dump(R/'evidencias/ANDAMENTO_CONTROLES.json',{'status':'EXECUTANDO','completed':done,'last':row});print('CONTROLE_CONCLUIDO',kind,b,method,round(result['macro_f1'],4),flush=True)
            # Bootstrap comum pareado só nas seis representações quando há referência.
            boot=intervalos(d.iloc[te],{m:probs[m] for m in METHODS},baseline if len(baseline)==6 else None)
            for contrast in boot['diferencas_pareadas_exploratorias']:contrast['contrast']='treino pareado menos treino completo, mesmo teste selecionado, seed42 com grupos atuais'
            boot['conditioning']='Grupos noticiários; fontes avaliadas fixas. Não mede variação entre possíveis sites de teste.'
            dump(out/'bootstrap_teste.json',boot)
    for b,d in data.items():
        meta=pd.read_csv(R/'dados'/(b+'_metadados.csv'),keep_default_na=False,dtype={'doc_id':str});assert meta.doc_id.tolist()==d.doc_id.tolist()
        if not meta.source.ne('').all():continue
        for seed in [42,43,44,45,46]:
            tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(d,groups=d.group_id));train=meta.iloc[tr];test=meta.iloc[te];major=int(train.label.value_counts().idxmax());mapping={site:int(frame.label.value_counts().idxmax()) for site,frame in train.groupby('source')};pred=test.source.map(mapping).fillna(major).to_numpy(dtype=int)
            out=R/'resultados/diagnostico_fonte'/b;out.mkdir(parents=True,exist_ok=True);saved=test[['doc_id','group_id','source','label']].copy();saved['prediction']=pred;saved['source_seen_in_train']=test.source.isin(mapping);saved.to_csv(out/f'seed_{seed}_predicoes.csv',index=False)
            diagnostics.append({'base':b,'split_seed':seed,'macro_f1':float(f1_score(test.label,pred,average='macro')),'accuracy':float(accuracy_score(test.label,pred)),'n_train':len(train),'n_test':len(test),'unseen_source_docs':int(sum(~test.source.isin(mapping))),'fitted_on_train_only':True})
    pd.DataFrame(diagnostics).to_csv(R/'DIAGNOSTICO_FONTES.csv',index=False);dump(R/'evidencias/ANDAMENTO_CONTROLES.json',{'status':'CONCLUIDO','fits':done,'representation_fits':sum(r['method'] in METHODS for r in rows),'length_diagnostic_fits':sum(r['method']=='comprimento' for r in rows),'source_majority_evaluations':len(diagnostics),'pending':'transferência depende dos modelos seed42 concluídos'})
    print('CONTROLES_FINALIZADOS',done,flush=True)
def run_transfer():
    import itertools,shutil
    data={b:load_data(b) for b in BASES[:3]};rows=[]
    audit=json.loads((R/'evidencias/AUDITORIA_TRANSFERENCIA.json').read_text(encoding='utf8'));assert audit['status']=='CONGELADO_ANTES_DE_PREDICOES'
    for src,tgt in itertools.permutations(BASES[:3],2):
        out=R/'resultados/transferencia'/f'{src}_para_{tgt}';sel=json.loads((out/'SELECAO.json').read_text(encoding='utf8'))
        if not sel['eligible']:continue
        part=pd.read_csv(out/'ALVO_CONGELADO.csv',dtype={'doc_id':str});d=data[tgt];assert part.doc_id.tolist()==d.doc_id.tolist() and part.evaluate.dtype==bool;te=np.flatnonzero(part.evaluate);test=d.iloc[te];probs={};baseline={}
        for m in METHODS:
            origin=R/'resultados/repeticoes'/src/'seed_42';rp=origin/(m+'_resultado.json');assert rp.exists(),f'Aguardar seed42 de {src}/{m}'
            fit=json.loads(rp.read_text(encoding='utf8'));model=XGBClassifier();model.load_model(origin/(m+'_xgboost.json'))
            if m in METHODS[:3]:X=load_matrix(fit['matrices']['target_'+tgt])[te]
            else:
                specs=[json.loads((E/'vetores_contextuais'/b/m/'spec.json').read_text(encoding='utf8')) for b in [src,tgt]]
                for key in ['model','max_length','pool','aggregation','precision']:assert specs[0][key]==specs[1][key],(m,key)
                X=np.load(E/'vetores_contextuais'/tgt/m/'all.npy',mmap_mode='r',allow_pickle=False)[te]
            prob=model.predict_proba(X)[:,1];pred=(prob>=.5).astype(int);probs[m]=prob;save=test[['doc_id','group_id','label']].copy();save['prob_fake']=prob;save['prediction']=pred;save.to_csv(out/(m+'_predicoes.csv'),index=False);shutil.copy2(origin/(m+'_xgboost.json'),out/(m+'_modelo_fonte.json'))
            ref=pd.read_csv(R/'resultados/repeticoes'/tgt/'seed_42'/(m+'_predicoes.csv'),dtype={'doc_id':str}).set_index('doc_id').loc[test.doc_id];assert ref.label.tolist()==test.label.tolist();baseline[m]=ref.prediction.to_numpy()
            row={'source':src,'target':tgt,'method':m,'n_source_train':fit['n_train'],'n_target_test':len(te),'macro_f1':float(f1_score(test.label,pred,average='macro')),'accuracy':float(accuracy_score(test.label,pred)),'internal_target_macro_f1_same_test':float(f1_score(test.label,ref.prediction,average='macro')),'excluded_target_test_docs':sel['excluded_test_docs'],'source_model_sha256':sha(origin/(m+'_xgboost.json')),'predictions_sha256':sha(out/(m+'_predicoes.csv'))}
            row['delta_external_minus_internal']=row['macro_f1']-row['internal_target_macro_f1_same_test'];dump(out/(m+'_resultado.json'),row);rows.append(row);pd.DataFrame(rows).to_csv(R/'RESULTADOS_TRANSFERENCIA.csv',index=False);print('TRANSFERENCIA',src,tgt,m,round(row['macro_f1'],4),flush=True);del X,model;gc.collect()
        boot=intervalos(test,probs,baseline)
        for contrast in boot['diferencas_pareadas_exploratorias']:contrast['contrast']='modelo treinado na fonte menos modelo treinado no alvo, mesmas notícias e grupos atuais, seed42'
        boot['conditioning']='Grupos noticiários e corpus/fontes de teste fixos; não mede variação entre possíveis corpora.'
        dump(out/'bootstrap_teste.json',boot)
    assert len(rows)==36;dump(R/'evidencias/ANDAMENTO_TRANSFERENCIA.json',{'status':'CONCLUIDO','evaluations':36,'new_fits':0,'target_adaptation':False,'source_models':'reutilizados de repeticoes/seed42','bootstrap_note':'Referência atual seed42 treinada no alvo, com grupos reforçados do Fake.Br; intervalos pareados são externos menos internos, sem causalidade isolada.'})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--transfer',action='store_true');a=p.parse_args()
    try:run_transfer() if a.transfer else run_controls()
    except Exception as e:dump(R/'evidencias'/('FALHA_TRANSFERENCIA.json' if a.transfer else 'FALHA_CONTROLES.json'),{'error':str(e),'traceback':traceback.format_exc()});raise
