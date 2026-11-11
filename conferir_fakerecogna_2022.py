from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
import json
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier


METHODS = ('tfidf', 'word2vec', 'fasttext', 'bertimbau', 'e5', 'qwen3', 'comprimento')


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_text(encoding='utf8'))


def unpack(root, path):
    with ZipFile(path) as z:
        assert z.testzip() is None, path
        manifest = json.loads(z.read('SHA256.json'))
        assert set(z.namelist()) == set(manifest) | {'SHA256.json'}
        for name, expected in manifest.items():
            target = (root / name).resolve()
            assert target.is_relative_to(root.resolve())
            payload = z.read(name)
            assert sha256(payload).hexdigest() == expected
            if name == 'resultados/AMBIENTE.json':
                target = root / 'resultados/AMBIENTE_COLAB.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


def verify(root, public, remote=None):
    sys.path.insert(0, str(public))
    from apoio_fakerecogna_20261110 import avaliar_fonte, dividir

    cohort = pd.read_csv(root/'dados/fakerecogna.csv', keep_default_na=False)
    audit = load(root/'evidencias/AUDITORIA_ENTRADA.json')
    assert digest(root/'dados/fakerecogna.csv') == audit['data_sha256']
    assert len(cohort) == 11858 and cohort.group_id.nunique() == 11644
    remote_rows = {}
    if remote is not None:
        assert remote['status'] == 'VALIDACAO_CONTEXTUAL_15_PASS' and len(remote['resultados']) == 15
        remote_rows = {(r['method'], r['seed']): r for r in remote['resultados']}
        assert set(remote_rows) == {(m,s) for m in METHODS[3:6] for s in range(42,47)}
    rows, diagnostics, parts = [], [], []
    for seed in range(42, 47):
        tr, te = dividir(root, cohort, seed)
        train = cohort.iloc[tr]
        test = cohort.iloc[te]
        parts.append({'seed': seed, 'train': len(tr), 'test': len(te),
                      'train_labels': train.label.value_counts().to_dict(),
                      'test_labels': test.label.value_counts().to_dict()})
        folder = root / f'resultados/seed_{seed}'
        avaliar_fonte(root, cohort, tr, te, seed)
        site = load(folder/'regra_site.json')
        for method in METHODS:
            prefix = folder / method
            if (method, seed) in remote_rows:
                result = remote_rows[method, seed]
                assert result['n_train'] == len(tr) and result['n_test'] == len(te)
                assert result['embeddings_sha256'] == remote['modelos'][method]['embeddings_sha256']
                assert sum(sum(row) for row in result['confusion_matrix']) == len(te)
                rows.append(result)
                continue
            result = load(prefix.with_name(prefix.name+'_resultado.json'))
            predfile = prefix.with_name(prefix.name+'_predicoes.csv')
            assert digest(predfile) == result['predictions_sha256']
            pred = pd.read_csv(predfile)
            for field in ['doc_id', 'group_id', 'label', 'canonical_sha256']:
                assert pred[field].tolist() == test[field].tolist(), (method, seed, field)
            assert result['n_train'] == len(tr) and result['n_test'] == len(te)
            assert result['seed'] == seed and result['method'] == method
            restored = XGBClassifier()
            restored.load_model(prefix.with_name(prefix.name+'_xgboost.json'))
            matrix = load_npz(prefix.with_name(prefix.name+'_test.npz')) if method == 'tfidf' else np.load(prefix.with_name(prefix.name+'_test.npy'), allow_pickle=False)
            p = restored.predict_proba(matrix)[:, 1]
            assert np.allclose(p, pred.prob_fake, atol=2e-6, rtol=0), (method, seed)
            assert np.array_equal(p >= .5, pred.prediction), (method, seed)
            metrics = classification_report(pred.label, pred.prediction, labels=[0, 1], target_names=['verdadeira', 'falsa'], output_dict=True, zero_division=0)
            assert np.isclose(f1_score(pred.label, pred.prediction, average='macro'), result['macro_f1'])
            assert np.isclose(accuracy_score(pred.label, pred.prediction), result['accuracy'])
            assert confusion_matrix(pred.label, pred.prediction, labels=[0, 1]).tolist() == result['confusion_matrix']
            for cls in ('verdadeira', 'falsa'):
                for metric in ('precision', 'recall', 'f1-score'):
                    assert np.isclose(metrics[cls][metric], result['classification_report'][cls][metric])
            assert result['bootstrap95']['valid_repetitions'] == 2000
            if method == 'comprimento':
                diagnostics.append(dict(seed=seed, length_macro_f1=result['macro_f1'], **site))
            else:
                rows.append(result)
        print('Divisão conferida:',seed, flush=True)
    assert len(rows) == 30 and len(diagnostics) == 5
    models = {}
    for method in METHODS[3:6]:
        if remote is not None:
            models[method] = remote['modelos'][method]
            spec = models[method]
            assert spec['shape'] == [len(cohort), spec['dim']] and spec['data_sha256'] == audit['data_sha256']
            continue
        folder = root/'vetores'/method
        spec = load(folder/'spec.json')
        p = folder/'all.npy'
        assert digest(p) == spec['embeddings_sha256']
        a = np.load(p, mmap_mode='r', allow_pickle=False)
        assert a.shape == (len(cohort), spec['dim']) and np.isfinite(a).all()
        for seed in range(42, 47):
            _, te = dividir(root, cohort, seed)
            assert np.array_equal(a[te], np.load(root/f'resultados/seed_{seed}/{method}_test.npy', allow_pickle=False))
        models[method] = spec
    summaries = {m: {'macro_f1': mean(r['macro_f1'] for r in rows if r['method'] == m),
                     'sd': stdev(r['macro_f1'] for r in rows if r['method'] == m),
                     'accuracy': mean(r['accuracy'] for r in rows if r['method'] == m)} for m in METHODS[:6]}
    domains = {str(k): dict(Counter(cohort.loc[cohort.label.eq(k), 'source_domain']).most_common(8)) for k in (0, 1)}
    diagnosis_mean = {k: mean(r[k] for r in diagnostics) for k in ('length_macro_f1', 'macro_f1', 'majority_macro_f1', 'source_rule_coverage')}
    summary = {'status': 'CONFERIDO', 'validation_location': '20 local + 15 no Colab' if remote else '35 local',
               'results': summaries, 'diagnostics_mean': diagnosis_mean,
               'diagnostics': diagnostics, 'partitions': parts, 'models': models,
               'most_common_domains': domains, 'individual_results': rows}
    target = root/'evidencias/VALIDACAO_FINAL_FAKERECOGNA_2022.json'
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summaries, indent=2), flush=True)
    print(json.dumps(diagnosis_mean, indent=2), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('public', type=Path)
    parser.add_argument('archives', type=Path, nargs='*')
    parser.add_argument('--contextual-audit', type=Path)
    args = parser.parse_args()
    for path in args.archives:
        unpack(args.root, path)
    verify(args.root, args.public, load(args.contextual_audit) if args.contextual_audit else None)
