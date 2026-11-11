"""Executa o piloto ou o experimento completo do FakeRecogna de 2022."""
from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from apoio_fakerecogna_20261110 import avaliar, carregar, consolidar, contextual, dividir, tokenizar, tradicionais
from reconstituir_fakerecogna import reconstituir


def executar(raiz, completo=False):
    reconstituir(raiz)
    dados = carregar(raiz)
    print(f'{len(dados)} textos, {dados.group_id.nunique()} grupos. NumPy {np.__version__}.', flush=True)
    if completo:
        tradicionais(raiz, dados)
        for metodo in ('bertimbau', 'e5', 'qwen3'):
            contextual(raiz, dados, metodo)
        consolidar(raiz, final=True)
        return

    treino, teste = dividir(raiz, dados, 42)
    inicio = time.perf_counter()
    tfidf = TfidfVectorizer(tokenizer=tokenizar, token_pattern=None, lowercase=False,
                           max_features=5000, ngram_range=(1, 2), max_df=.85, min_df=2,
                           dtype=np.float32)
    treino_tfidf = tfidf.fit_transform(dados.iloc[treino].text)
    teste_tfidf = tfidf.transform(dados.iloc[teste].text)
    avaliar(raiz, dados, treino, teste, 'tfidf', treino_tfidf, teste_tfidf,
            {'seed': 42, 'fitted_on_train_only': True, 'vocabulary_size': len(tfidf.vocabulary_),
             'representation_seconds': time.perf_counter() - inicio})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--raiz', type=Path, default=Path('execucao_fakerecogna'))
    parser.add_argument('--completo', action='store_true')
    args = parser.parse_args()
    executar(args.raiz, args.completo)
