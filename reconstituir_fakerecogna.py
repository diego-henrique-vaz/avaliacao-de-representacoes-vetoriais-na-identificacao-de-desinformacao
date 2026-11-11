"""Obtém a versão oficial e recompõe a entrada a partir dos índices congelados."""
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlretrieve
import argparse
import hashlib
import json
import re
import unicodedata

import pandas as pd


MATERIAL = Path(__file__).resolve().parent


def sha(arquivo):
    return hashlib.sha256(arquivo.read_bytes()).hexdigest()


def dominio(url):
    try:
        return urlsplit(url).netloc.lower().removeprefix('www.')
    except ValueError:
        return ''


def hash_texto(texto):
    canonico = ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', texto).lower()))
    return hashlib.sha256(canonico.encode()).hexdigest()


def reconstituir(raiz, fonte=None):
    raiz = Path(raiz)
    auditoria = json.loads((MATERIAL / 'auditoria_fakerecogna_2022.json').read_text(encoding='utf8'))
    if fonte is None:
        fonte = raiz / 'dados_brutos/FakeRecogna_no_removal_words.xlsx'
        fonte.parent.mkdir(parents=True, exist_ok=True)
        if not fonte.exists():
            temporario = fonte.with_suffix('.tmp')
            urlretrieve(auditoria['source_url'], temporario)
            if sha(temporario) != auditoria['source_sha256']:
                raise ValueError('O arquivo baixado não corresponde à revisão citada.')
            temporario.replace(fonte)
    fonte = Path(fonte)
    if sha(fonte) != auditoria['source_sha256']:
        raise ValueError('Confira a revisão do arquivo oficial do FakeRecogna.')
    original = pd.read_excel(fonte).fillna('')
    mapa = pd.read_csv(MATERIAL / 'particoes_fakerecogna_2022.csv')
    selecionado = original.iloc[mapa.row_original].reset_index(drop=True)
    saida = mapa[['doc_id', 'row_original']].copy()
    saida['text'] = selecionado.Noticia.astype(str)
    saida['label'] = 1 - selecionado.Classe.astype(int)
    saida['url'] = selecionado.URL.astype(str).str.strip()
    saida['source_date'] = selecionado.Data.astype(str)
    saida['category'] = selecionado.Categoria.astype(str)
    saida['source_domain'] = saida.url.map(dominio)
    saida['canonical_sha256'] = saida.text.map(hash_texto)
    saida['words'] = saida.text.str.split().str.len()
    saida['group_id'] = mapa.group_id
    if not saida.label.equals(mapa.label) or not saida.canonical_sha256.equals(mapa.canonical_sha256):
        raise ValueError('O conteúdo reconstruído diverge das classes ou dos textos congelados.')
    dados = raiz / 'dados'
    evidencias = raiz / 'evidencias'
    dados.mkdir(parents=True, exist_ok=True)
    evidencias.mkdir(parents=True, exist_ok=True)
    tabela = dados / 'fakerecogna.csv'
    saida.to_csv(tabela, index=False, lineterminator='\r\n')
    if sha(tabela) != auditoria['data_sha256']:
        raise ValueError('O CSV não corresponde ao arquivo utilizado no experimento.')
    for semente in range(42, 47):
        particao = mapa[['doc_id', 'group_id', 'label', 'canonical_sha256']].copy()
        particao['split'] = mapa[f'seed_{semente}']
        particao.to_csv(dados / f'particoes_seed_{semente}.csv', index=False, lineterminator='\r\n')
    (evidencias / 'AUDITORIA_ENTRADA.json').write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding='utf8')
    return tabela


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raiz', type=Path, default=Path('execucao_fakerecogna'))
    parser.add_argument('--fonte', type=Path)
    args = parser.parse_args()
    print(reconstituir(args.raiz, args.fonte))
