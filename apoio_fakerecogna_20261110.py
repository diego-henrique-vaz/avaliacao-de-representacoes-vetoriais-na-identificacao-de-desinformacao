"""Experimento separado do FakeRecogna; não modifica os resultados anteriores."""
from pathlib import Path
import re,json,hashlib,unicodedata,time,gc,platform,importlib.metadata,os,zipfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
from xgboost import XGBClassifier
from gensim.models import Word2Vec,FastText
from scipy.sparse import save_npz,issparse

PARAMS=dict(objective='binary:logistic',tree_method='hist',n_estimators=300,max_depth=3,learning_rate=.05,subsample=1.,colsample_bytree=1.,random_state=42,n_jobs=2)
MODELS={
 'bertimbau':dict(id='neuralmind/bert-base-portuguese-cased',revision='94d69c95f98f7d5b2a8700c420230ae10def0baa',dim=768,prefix=''),
 'e5':dict(id='intfloat/multilingual-e5-small',revision='614241f622f53c4eeff9890bdc4f31cfecc418b3',dim=384,prefix='query: '),
 'qwen3':dict(id='Qwen/Qwen3-Embedding-0.6B',revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',dim=1024,prefix='Instruct: Represent the linguistic and semantic content of this Portuguese news article for news veracity classification.\nQuery:')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8');tmp.replace(p)
def tokenizar(t):return re.findall(r'(?u)\b\w\w+\b',unicodedata.normalize('NFC',t).lower())
def stable_hash(w):return int.from_bytes(hashlib.sha256(w.encode()).digest()[:8],'little')
def carregar(root):
    root=Path(root);d=pd.read_csv(root/'dados/fakerecogna.csv',keep_default_na=False);audit=json.loads((root/'evidencias/AUDITORIA_ENTRADA.json').read_text(encoding='utf8'))
    assert sha(root/'dados/fakerecogna.csv')==audit['data_sha256']
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','gensim','xgboost','torch','transformers']}
    versions.update(python=platform.python_version(),platform=platform.platform())
    dump(root/'resultados/AMBIENTE.json',versions)
    return d
def dividir(root,d,seed):
    p=pd.read_csv(Path(root)/f'dados/particoes_seed_{seed}.csv',keep_default_na=False)
    assert p.doc_id.tolist()==d.doc_id.tolist() and p.group_id.tolist()==d.group_id.tolist() and p.label.tolist()==d.label.tolist()
    tr=np.flatnonzero(p.split.eq('train'));te=np.flatnonzero(p.split.eq('test'))
    assert not set(d.iloc[tr].group_id)&set(d.iloc[te].group_id)
    assert not set(d.iloc[tr].canonical_sha256)&set(d.iloc[te].canonical_sha256)
    return tr,te
def intervalo(test,pred):
    codes,groups=pd.factorize(test.group_id,sort=True);cells=np.zeros((len(groups),4),np.int32)
    np.add.at(cells,(codes,test.label.to_numpy()*2+pred),1)
    rng=np.random.default_rng(42);samples=[]
    for _ in range(2000):
        w=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups));tn,fp,fn,tp=w@cells
        if tn+fp==0 or fn+tp==0:continue
        samples.append((2*tp/max(2*tp+fp+fn,1)+2*tn/max(2*tn+fp+fn,1))/2)
    return {'lower':float(np.quantile(samples,.025)),'upper':float(np.quantile(samples,.975)),'valid_repetitions':len(samples),'sampling':'grupos inteiros com reposição; classificador fixo; não abrange novos treinos/fontes'}
def avaliar(root,d,tr,te,method,a,b,extra):
    out=Path(root)/f'resultados/seed_{extra["seed"]}';out.mkdir(parents=True,exist_ok=True)
    sig=hashlib.sha256(json.dumps({'dataset':sha(Path(root)/'dados/fakerecogna.csv'),'train':d.iloc[tr].doc_id.tolist(),'test':d.iloc[te].doc_id.tolist(),'method':method,'params':PARAMS},sort_keys=True).encode()).hexdigest()
    rp=out/(method+'_resultado.json')
    if rp.exists():
        result=json.loads(rp.read_text(encoding='utf8'));assert result['signature']==sig and sha(out/(method+'_predicoes.csv'))==result['predictions_sha256'];print('RETOMADO',extra['seed'],method,result['macro_f1'],flush=True);return result
    model=XGBClassifier(**PARAMS);start=time.perf_counter();model.fit(a,d.iloc[tr].label.to_numpy());secs=time.perf_counter()-start
    probs=model.predict_proba(b)[:,1];pred=(probs>=.5).astype(int);test=d.iloc[te];r=test[['doc_id','group_id','canonical_sha256','label']].copy();r['prob_fake']=probs;r['prediction']=pred;r.to_csv(out/(method+'_predicoes.csv'),index=False)
    model.save_model(out/(method+'_xgboost.json'))
    if issparse(b):save_npz(out/(method+'_test.npz'),b)
    else:np.save(out/(method+'_test.npy'),b)
    result={'method':method,'signature':sig,'n_train':len(tr),'n_test':len(te),'params':PARAMS,'fit_seconds':secs,'macro_f1':float(f1_score(test.label,pred,average='macro')),'accuracy':float(accuracy_score(test.label,pred)),'classification_report':classification_report(test.label,pred,labels=[0,1],target_names=['verdadeira','falsa'],output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(test.label,pred,labels=[0,1]).tolist(),'bootstrap95':intervalo(test,pred),'predictions_sha256':sha(out/(method+'_predicoes.csv')),**extra}
    dump(rp,result);print('CONCLUÍDO',extra['seed'],method,round(result['macro_f1'],4),flush=True);return result
def tradicionais(root,d,seeds=range(42,47)):
    for seed in seeds:
        tr,te=dividir(root,d,seed);train=d.iloc[tr];test=d.iloc[te]
        for method in ['tfidf','word2vec','fasttext']:
            if (Path(root)/f'resultados/seed_{seed}/{method}_resultado.json').exists():continue
            start=time.perf_counter();meta={'seed':seed,'fitted_on_train_only':True}
            if method=='tfidf':
                v=TfidfVectorizer(tokenizer=tokenizar,token_pattern=None,preprocessor=None,lowercase=False,max_features=5000,ngram_range=(1,2),max_df=.85,min_df=2,dtype=np.float32)
                a=v.fit_transform(train.text);b=v.transform(test.text);meta['vocabulary_size']=len(v.vocabulary_)
            else:
                pt=[tokenizar(t) for t in train.text];pe=[tokenizar(t) for t in test.text]
                args=dict(sentences=pt,vector_size=100,window=5,min_count=1,workers=1,sg=1,seed=42,epochs=10,negative=5,hs=0,sample=.001,ns_exponent=.75,alpha=.025,min_alpha=.0001,hashfxn=stable_hash)
                m=FastText(**args,min_n=3,max_n=6,bucket=200000) if method=='fasttext' else Word2Vec(**args)
                def transform(docs):
                    vectors=[]
                    for doc in docs:
                        vs=[m.wv[w] for w in doc if method=='fasttext' or w in m.wv.key_to_index];vectors.append(np.mean(vs,axis=0) if vs else np.zeros(100))
                    return np.asarray(vectors,dtype=np.float32)
                a,b=transform(pt),transform(pe);total=sum(map(len,pe));meta['test_vocabulary_coverage']=sum(w in m.wv.key_to_index for doc in pe for w in doc)/max(total,1);meta['test_zero_vectors']=int(np.sum(~np.any(b,axis=1)));del m
            meta['representation_seconds']=time.perf_counter()-start;avaliar(root,d,tr,te,method,a,b,meta);del a,b;gc.collect()
        vals=np.log1p(d.words.to_numpy(dtype=np.float32)).reshape(-1,1)
        avaliar(root,d,tr,te,'comprimento',vals[tr],vals[te],{'seed':seed,'single_feature':'log1p(word_count)'})
        avaliar_fonte(root,d,tr,te,seed)
    consolidar(root,final=False)
def extrair(root,d,method):
    import torch
    from transformers import AutoTokenizer,AutoModel
    from tqdm.auto import tqdm
    from apoio_contextuais_simples import dividir_texto,resumir_saida
    assert torch.cuda.is_available(),'GPU indisponível: preservar os resultados e habilitar T4 antes da extração contextual.'
    cfg=MODELS[method];folder=Path(root)/'vetores'/method;folder.mkdir(parents=True,exist_ok=True);fp=sha(Path(root)/'dados/fakerecogna.csv');final=folder/'all.npy'
    spec={**cfg,'data_sha256':fp,'limit_tokens':512,'pool':'content_mean' if method=='bertimbau' else 'attention_mean' if method=='e5' else 'last_valid_token','aggregation':'chunk L2(E5/Qwen); content-token weighted mean; document L2(E5/Qwen)','precision':'float16','device':torch.cuda.get_device_name(0)}
    if final.exists():
        old=json.loads((folder/'spec.json').read_text(encoding='utf8'));assert all(old[k]==v for k,v in spec.items());assert sha(final)==old['embeddings_sha256'];return np.load(final,mmap_mode='r',allow_pickle=False)
    tok=AutoTokenizer.from_pretrained(cfg['id'],revision=cfg['revision']);tok.padding_side='left' if method=='qwen3' else 'right'
    start=time.perf_counter();blocks=[];owners=[]
    for i,t in enumerate(d.text):
        for block in dividir_texto(tok,t,cfg['prefix'],512):blocks.append(block);owners.append(i)
    model=AutoModel.from_pretrained(cfg['id'],revision=cfg['revision'],torch_dtype=torch.float16,attn_implementation='eager').cuda().eval()
    for p in model.parameters():p.requires_grad=False
    sums=np.zeros((len(d),cfg['dim']),dtype=np.float64);weights=np.zeros(len(d),dtype=np.float64);begin=0
    progress=folder/'andamento.json';checkpoint=folder/'checkpoint.npz'
    signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as cp:
            assert str(cp['signature'].item())==signature
            sums=cp['sums'];weights=cp['weights'];begin=int(cp['blocks_done'].item())
    batch=8 if method=='qwen3' else 32
    try:
        for j in tqdm(range(begin,len(blocks),batch),desc=method):
            group=blocks[j:j+batch];inputs=tok.pad([{k:b[k] for k in ['input_ids','attention_mask']} for b in group],padding=True,return_tensors='pt');length=inputs['input_ids'].shape[1];masks=[]
            for b in group:
                zeros=[0]*(length-len(b['conteudo']));masks.append(zeros+b['conteudo'] if tok.padding_side=='left' else b['conteudo']+zeros)
            inputs={k:v.cuda() for k,v in inputs.items()};content=torch.tensor(masks,device='cuda')
            with torch.inference_mode():
                hidden=model(**inputs).last_hidden_state;vectors=resumir_saida(hidden,inputs['attention_mask'],content,method).float()
                if method!='bertimbau':vectors=torch.nn.functional.normalize(vectors,p=2,dim=1)
                vectors=vectors.cpu().numpy()
            for k,b in enumerate(group):i=owners[j+k];sums[i]+=vectors[k]*b['peso'];weights[i]+=b['peso']
            end=j+len(group)
            if end% (batch*20)==0 or end==len(blocks):
                temporary=checkpoint.with_suffix('.tmp')
                with temporary.open('wb') as f:np.savez(f,sums=sums,weights=weights,blocks_done=np.asarray(end),signature=np.asarray(signature))
                temporary.replace(checkpoint);dump(progress,{'spec':spec,'blocks_done':end,'blocks_total':len(blocks)})
        assert np.all(weights>0)
        a=(sums/weights[:,None]).astype(np.float32)
        if method!='bertimbau':a=a/np.linalg.norm(a,axis=1,keepdims=True)
        assert np.isfinite(a).all();np.save(final,a);spec.update(embeddings_sha256=sha(final),shape=list(a.shape),extraction_seconds=time.perf_counter()-start,blocks=len(blocks));dump(folder/'spec.json',spec)
    finally:del model;gc.collect();torch.cuda.empty_cache()
    return np.load(final,mmap_mode='r',allow_pickle=False)
def contextual(root,d,method,seeds=range(42,47)):
    arr=extrair(root,d,method)
    for seed in seeds:
        tr,te=dividir(root,d,seed);avaliar(root,d,tr,te,method,arr[tr],arr[te],{'seed':seed,'encoder_frozen':True,'model':MODELS[method],'embeddings_sha256':sha(Path(root)/'vetores'/method/'all.npy')})
    consolidar(root,final=False)
def consolidar(root,final=True):
    rows=[]
    for p in (Path(root)/'resultados').glob('seed_*/*_resultado.json'):
        r=json.loads(p.read_text(encoding='utf8'));rows.append({k:r[k] for k in ['seed','method','macro_f1','accuracy','n_train','n_test','fit_seconds']})
    df=pd.DataFrame(rows).sort_values(['method','seed']);df.to_csv(Path(root)/'resultados/RESULTADOS_INDIVIDUAIS.csv',index=False)
    summary=df.groupby('method').agg(mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),mean_accuracy=('accuracy','mean'),repetitions=('seed','size')).reset_index();summary.to_csv(Path(root)/'resultados/RESUMO.csv',index=False)
    if final:assert len(df)==35 and set(df.method)=={'tfidf','word2vec','fasttext','bertimbau','e5','qwen3','comprimento'}
    dump(Path(root)/'resultados/ANDAMENTO.json',{'status':'CONCLUIDO' if final else 'PARCIAL','conditions_done':len(rows),'expected_main_fits':30,'expected_length_fits':5,'site_rules':len(list((Path(root)/'resultados').glob('seed_*/regra_site.json'))),'new_dataset':'FakeRecogna original; outros resultados preservados'})
    print(summary.to_string(index=False),flush=True);return summary
def pacote(root,stage):
    root=Path(root);p=root.parent/('TG_FAKERECOGNA_'+stage+'.zip');manifest={}
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in root.rglob('*'):
            if f.is_file() and f.suffix not in ['.tmp','.pyc'] and f.name not in ['sums.npy','weights.npy','checkpoint.npz']:
                n=f.relative_to(root).as_posix();z.write(f,'TG_FAKERECOGNA/'+n);manifest[n]={'bytes':f.stat().st_size,'sha256':sha(f)}
        z.writestr('TG_FAKERECOGNA/MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    return str(p)


def avaliar_fonte(root, dados, treino, teste, semente):
    """Aprende a maioria de cada domínio no treino, usando a maioria global como reserva."""
    dados_treino = dados.iloc[treino]
    dados_teste = dados.iloc[teste]
    tem_dominio = dados_treino.source_domain.ne('')
    maioria = int(dados_treino.label.value_counts().idxmax())
    mapa = dados_treino.loc[tem_dominio].groupby('source_domain').label.agg(
        lambda classes: int(classes.value_counts().idxmax())
    )
    dominios_teste = dados_teste.source_domain
    conhecidos = dominios_teste.isin(mapa.index)
    pred = dominios_teste.map(mapa).fillna(maioria).to_numpy(dtype=int)
    caminho = Path(root) / f'resultados/seed_{semente}/regra_site.json'
    dump(caminho, {
        'macro_f1': float(f1_score(dados_teste.label, pred, average='macro')),
        'accuracy': float(accuracy_score(dados_teste.label, pred)),
        'majority_macro_f1': float(f1_score(dados_teste.label, np.full(len(teste), maioria), average='macro')),
        'n_test': len(teste),
        'known_site_documents': int(conhecidos.sum()),
        'missing_site_documents': int(dominios_teste.eq('').sum()),
        'unseen_site_documents': int((~conhecidos & dominios_teste.ne('')).sum()),
        'source_rule_coverage': float(conhecidos.mean()),
    })
