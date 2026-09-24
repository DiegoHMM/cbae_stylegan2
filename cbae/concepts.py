import random
import torch
import torch.nn.functional as F

from .config import N_CLASSES, HAIR, OTHER_HAIR, IGNORE

def concept_slice(k):
    s = sum(N_CLASSES[:k])
    return slice(s, s + N_CLASSES[k])


def valid_targets(k):
    """Classes que podem ser alvo de intervenção no conceito k ("Other Hair" não tem significado visual)."""
    return [v for v in range(N_CLASSES[k]) if not (k == HAIR and v == OTHER_HAIR)]

def sample_target(k):
    """ Sorteia a classe do conceito k que será alterada """
    return random.choice(valid_targets(k))

def masked_cross_entropy(logits, y):
    '''Cross entropy entre os logits de um conceito e os pseudo-rótulos de M,
    ignorando as amostras marcadas com IGNORE (confiança de M abaixo do limiar).

    logits: [B, n_classes]   bloco do conceito em c
    y:      [B]              classe de cada amostra, ou IGNORE
    '''
    if (y != IGNORE).any(): # se pelo menos um rotulo valido
        return F.cross_entropy(logits, y, ignore_index=IGNORE)
    return logits.sum() * 0.0 #nenhum rotulo valido

def concept_cross_entropy(c, labels):
    """Soma da masked_cross_entropy dos 11 blocos de conceito de c.
    As 40 dimensões unknown não entram (ficam livres para a reconstrução).

    c:      [B, CONCEPT_DIM]  saída do encoder
    labels: lista com 1 tensor [B] por conceito (pseudo-rótulos de M, com IGNORE)
    """
    return sum(masked_cross_entropy(c[:, concept_slice(k)], labels[k]) for k in range(len(N_CLASSES)))


@torch.no_grad()
def swap_intervene(c, k, v):
    """Intervenção no conceito k: troca o maior logit do bloco com o logit da classe v.

    c: [B, CONCEPT_DIM]   saída do encoder
    k: int                índice do conceito
    v: int ou tensor [B]  classe alvo (a mesma para o batch ou uma por imagem)
    Retorna uma cópia de c; o c original não é alterado.
    """
    c_new = c.clone()
    concept_block = c_new[:, concept_slice(k)]                      # view de c_new: [B, n_classes]
    i_max_class = concept_block.argmax(dim=1)                       # classe atual de cada imagem
    rows = torch.arange(concept_block.size(0), device=c.device)

    old_v_classes_value = concept_block[rows, v].clone()            # guarda o valor de v antes de sobrescrever
    concept_block[rows, v] = concept_block[rows, i_max_class]       # v recebe o máximo
    concept_block[rows, i_max_class] = old_v_classes_value          # o antigo máximo recebe o valor de v
    return c_new