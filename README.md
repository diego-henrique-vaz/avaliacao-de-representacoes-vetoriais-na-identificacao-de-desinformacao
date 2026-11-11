# Avaliação de Representações Vetoriais na Identificação de Desinformação

Código do trabalho de graduação da Fatec Ourinhos. A pesquisa compara TF-IDF, Word2Vec, FastText, BERTimbau, E5 e Qwen3 Embedding com o mesmo classificador, XGBoost, em cinco conjuntos de língua portuguesa. As médias e desvios são calculados separadamente em cada conjunto, com cinco divisões por grupos e macro-F1 como medida principal.

## Executar o exemplo com FakeRecogna (2022)

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diego-henrique-vaz/avaliacao-de-representacoes-vetoriais-na-identificacao-de-desinformacao/blob/main/TG_FakeRecogna_2022.ipynb)

Execute as células em ordem. O notebook instala as dependências e chama `executar_fakerecogna_2022.py` em um novo processo Python para carregar as versões instaladas. O script baixa a planilha da revisão oficial, recompõe o recorte com `particoes_fakerecogna_2022.csv` e roda TF-IDF na divisão 42. Para repetir as seis representações nas cinco divisões, selecione GPU no Colab e altere `EXECUTAR_ESTUDO_COMPLETO` para `True`. A extração de BERTimbau, E5 e Qwen3 baixa os pesos públicos dos três modelos. Copie os resultados para um local persistente antes de encerrar a sessão.

O arquivo `auditoria_fakerecogna_2022.json` registra a origem, os hashes e os filtros. O mapa tem os índices originais, os grupos e as divisões, sem o corpo das notícias. O script `reconstituir_fakerecogna.py` também pode ser chamado na linha de comando. A saída é conferida pelo hash da entrada usada no experimento.

## Repetir as quatro avaliações anteriores

- [`TG_Robustez_Seis_Representacoes.ipynb`](TG_Robustez_Seis_Representacoes.ipynb) reproduz a classificação com os vetores congelados. A condição inicial é FakeTrue.Br na divisão 42. A opção de estudo completo usa os recortes e configurações do material de reprodução.
- [`TG_Extrair_Vetores_Contextuais.ipynb`](TG_Extrair_Vetores_Contextuais.ipynb) gera novos vetores com os modelos fixados. Os números do artigo foram obtidos com as matrizes congeladas. O equipamento e a precisão numérica podem afetar uma nova extração.
- Para essas quatro bases, extraia os quatro ZIPs que acompanham os fontes do artigo no projeto compartilhado do Overleaf. Os nomes começam com `TG_REPRODUCAO_CODIGOS_DADOS`, `TG_VETORES_BERTIMBAU`, `TG_VETORES_E5` e `TG_VETORES_QWEN3`. Ajuste `PASTA` para a raiz extraída. Esses pacotes contêm os recortes exatos, os mapas e as matrizes. O acesso a eles depende de permissão no projeto do artigo. Os links das bases abaixo permitem consultar os recursos de origem.

## Bases e recortes

| Base | Origem e recorte |
| --- | --- |
| Fake.Br | [Repositório oficial, revisão fixada](https://github.com/roneysco/Fake.br-Corpus/tree/780f5516c4ae070761632d98ac3368f3ded09d35). 7.200 textos integrais. Mantidos juntos os pares e demais registros agrupados. |
| BoatosBr | [Arquivo oficial, revisão fixada](https://raw.githubusercontent.com/Felipe-Harrison/boatos-br-corpus/afd1f31d204f9e992cb2f9fc582faaa2ba645898/base_simples/boatos_br_corpus_simples.json). 2.321 documentos com seleção de rótulos binários e categorias verificadas no artigo. |
| FakeTrue.Br | [Repositório oficial](https://github.com/jpchav98/FakeTrue.Br). 604 documentos de pares em que os dois textos têm entre 200 e 2.000 palavras, após as exclusões documentadas. |
| PT-EU | [Repositório oficial](https://github.com/ro-afonso/fake-news-pt-eu), de Afonso e Rosas (2024). Recorte de 2.000 documentos de português europeu. |
| FakeRecogna | [Garcia, Afonso e Papa (2022)](https://doi.org/10.1007/978-3-030-98305-5_6) e [planilha fixada](https://github.com/Gabriel-Lino-garcia/FakeRecogna/blob/58f8f6ebd419970527875ba243c9c891781747cd/dataset/FakeRecogna_no_removal_words.xlsx). 11.858 documentos após excluir 44 registros de grupos com proveniência ambígua. |

Cada conjunto tem critérios próprios de seleção, origem e rotulagem. As métricas medem acerto em relação às classes atribuídas pelos conjuntos. Consulte as tabelas de referência e a descrição dos controles para interpretar os resultados.

## Resultados de referência

[`resumo_repeticoes.csv`](resumo_repeticoes.csv) reúne a média e o desvio padrão amostral de macro-F1 de cada representação em cada base. As cinco divisões podem compartilhar documentos entre si. Para o FakeRecogna de 2022, [`resultados_fakerecogna_2022.csv`](resultados_fakerecogna_2022.csv) discrimina as 30 condições e [`diagnosticos_fakerecogna_2022.csv`](diagnosticos_fakerecogna_2022.csv) traz os controles de comprimento e fonte. Na condição de exemplo, o TF-IDF na divisão 42 obteve macro-F1 **0,9292** na execução de referência em Windows e **0,9258** no teste do notebook público em Colab com NumPy 2.2.6. Os números do artigo vêm dos ajustes registrados nos CSVs.

As versões e revisões dos extratores usados na quinta base estão em [`modelos_fakerecogna_2022.json`](modelos_fakerecogna_2022.json). O arquivo `conferir_fakerecogna_2022.py` verifica os artefatos gravados pela execução completa, recarregando cada modelo e recalculando as métricas a partir das predições.
