"""Parte específica dos três modelos pré-treinados. Não treina o classificador.

O notebook mostra as etapas comuns. Este arquivo trata apenas a divisão de
notícias longas e a leitura dos vetores devolvidos por cada modelo.
"""
from pathlib import Path
import gc, hashlib, json
import numpy as np


def dividir_texto(tokenizador, texto, prefixo='', limite=512):
    conteudo = tokenizador.encode(texto, add_special_tokens=False, truncation=False)
    inicio = tokenizador.encode(prefixo, add_special_tokens=False) if prefixo else []
    espaco = limite - len(inicio) - tokenizador.num_special_tokens_to_add(pair=False)
    if not conteudo or espaco <= 0:
        raise ValueError('Texto vazio ou instrução maior que o limite do bloco.')
    blocos = []
    for posicao in range(0, len(conteudo), espaco):
        trecho = conteudo[posicao:posicao + espaco]
        ids = tokenizador.build_inputs_with_special_tokens(inicio + trecho)
        # Marcadores -1 indicam posições de texto, sem confundir tokens especiais.
        estrutura = tokenizador.build_inputs_with_special_tokens([-1] * (len(inicio) + len(trecho)))
        posicoes = [i for i, valor in enumerate(estrutura) if valor == -1]
        mascara = [0] * len(ids)
        for i in posicoes[len(inicio):]:
            mascara[i] = 1
        assert len(ids) <= limite and sum(mascara) == len(trecho)
        blocos.append({'input_ids': ids, 'attention_mask': [1] * len(ids),
                       'conteudo': mascara, 'peso': len(trecho)})
    assert sum(b['peso'] for b in blocos) == len(conteudo)
    return blocos


def resumir_saida(saida, atencao, conteudo, metodo):
    import torch
    if metodo == 'qwen3':
        posicoes = torch.arange(atencao.shape[1], device=atencao.device).expand_as(atencao)
        ultima = posicoes.masked_fill(atencao == 0, -1).max(dim=1).values
        return saida[torch.arange(len(saida), device=saida.device), ultima]
    mascara = conteudo if metodo == 'bertimbau' else atencao
    mascara = mascara.unsqueeze(-1).to(saida.dtype)
    return (saida * mascara).sum(1) / mascara.sum(1).clamp(min=1)


def extrair_embeddings(textos, identificador, revisao, metodo, dimensao, prefixo,
                       pasta, lote=4, limite=512, salvar_a_cada=16):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from tqdm.auto import tqdm
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    precisao = torch.float16 if dispositivo == 'cuda' else torch.float32
    pasta = Path(pasta) / (metodo + '_' + dispositivo)
    pasta.mkdir(parents=True, exist_ok=True)
    # Impede continuar uma matriz de outro conjunto, ordem ou configuração.
    assinatura = hashlib.sha256(json.dumps(list(textos), ensure_ascii=False).encode()).hexdigest()
    configuracao = dict(textos_sha256=assinatura, modelo=identificador, revisao=revisao,
                        metodo=metodo, dimensao=dimensao, prefixo=prefixo,
                        limite=limite, precisao=str(precisao), lote=lote)
    registro = pasta / 'andamento.json'
    inicio = 0
    if registro.exists():
        anterior = json.loads(registro.read_text(encoding='utf8'))
        if anterior['configuracao'] != configuracao:
            raise ValueError('O checkpoint é de outra configuração. Use outra pasta.')
        inicio = anterior['documentos_prontos']
    if not 0 <= inicio <= len(textos):
        raise ValueError('Posição inválida no checkpoint.')
    caminho = pasta / 'vetores.npy'
    if inicio and not caminho.exists():
        raise ValueError('Checkpoint sem a matriz correspondente.')
    matriz = np.lib.format.open_memmap(caminho, mode='r+' if caminho.exists() else 'w+',
                                      dtype=np.float32, shape=(len(textos), dimensao))
    if matriz.shape != (len(textos), dimensao):
        raise ValueError('A matriz salva tem outro tamanho. Use outra pasta.')
    if inicio == len(textos):
        if not np.isfinite(matriz).all():
            raise ValueError('Matriz final incompleta.')
        return matriz
    tokenizador = AutoTokenizer.from_pretrained(identificador, revision=revisao)
    tokenizador.padding_side = 'left' if metodo == 'qwen3' else 'right'
    modelo = AutoModel.from_pretrained(identificador, revision=revisao,
                                      torch_dtype=precisao, attn_implementation='eager')
    modelo = modelo.to(dispositivo).eval()
    for parametro in modelo.parameters():
        parametro.requires_grad = False
    try:
        for i in tqdm(range(inicio, len(textos)), desc=metodo):
            blocos = dividir_texto(tokenizador, textos[i], prefixo, limite)
            vetores, pesos = [], []
            for j in range(0, len(blocos), lote):
                grupo = blocos[j:j + lote]
                entradas = tokenizador.pad(
                    [{k: b[k] for k in ['input_ids', 'attention_mask']} for b in grupo],
                    padding=True, return_tensors='pt')
                tamanho = entradas['input_ids'].shape[1]
                mascaras = []
                for b in grupo:
                    zeros = [0] * (tamanho - len(b['conteudo']))
                    mascaras.append(zeros + b['conteudo'] if tokenizador.padding_side == 'left'
                                    else b['conteudo'] + zeros)
                conteudo = torch.tensor(mascaras, device=dispositivo)
                entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
                with torch.inference_mode():
                    saida = modelo(**entradas).last_hidden_state
                    vetor = resumir_saida(saida, entradas['attention_mask'], conteudo, metodo).float()
                    if metodo != 'bertimbau':
                        vetor = torch.nn.functional.normalize(vetor, p=2, dim=1)
                vetores.extend(vetor.cpu().numpy())
                pesos.extend(b['peso'] for b in grupo)
            documento = np.average(np.asarray(vetores, dtype=np.float32), axis=0, weights=np.asarray(pesos))
            if metodo != 'bertimbau':
                documento = documento / np.linalg.norm(documento)
            if not np.isfinite(documento).all():
                raise ValueError('Vetor inválido no documento ' + str(i))
            matriz[i] = documento.astype(np.float32)
            if (i + 1) % salvar_a_cada == 0 or i + 1 == len(textos):
                matriz.flush()
                temporario = registro.with_suffix('.tmp')
                temporario.write_text(json.dumps({'configuracao': configuracao,
                                                 'documentos_prontos': i + 1}), encoding='utf8')
                temporario.replace(registro)
    finally:
        del modelo
        gc.collect()
        if dispositivo == 'cuda':
            torch.cuda.empty_cache()
    return matriz
