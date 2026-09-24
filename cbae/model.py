import torch
import torch.nn as nn

from .config import CONCEPT_DIM

class CBAE(nn.Module):
    def __init__(self, w_dim=512, concept_dim=CONCEPT_DIM, num_ws=14):
        super().__init__()
        self.encoder = mlp(w_dim, w_dim, concept_dim)
        self.decoder = mlp(concept_dim, w_dim, w_dim)
        self.num_ws = num_ws
        self.apply(init_weights)

    # w	[B, 14, 512]
    # w.mean(dim=1): média sobre as 14 cópias	[B, 512]
    # self.encoder(...): MLP 512 → 66	[B, 66] = c
    def enc(self, w):
        return self.encoder(w.mean(dim=1))

    # c [B, 66]
    # self.decoder(c): MLP 66 → 512  [B, 512]
    # .unsqueeze(1): cria uma dimensão de tamanho 1 na posição 1 [B, 1, 512]
    # .repeat(1, 14, 1): repete 1× o batch, 14× a dim 1 e 1× as 512 posições	[B, 14, 512]
    def dec(self, c):
        return self.decoder(c).unsqueeze(1).repeat(1, self.num_ws, 1)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def mlp(d_in, d_hid, d_out):
    layers = []
    d = d_in
    for _ in range(3):
        layers += [nn.Linear(d, d_hid), nn.LeakyReLU(0.1), nn.BatchNorm1d(d_hid)]
        d=d_hid
    return nn.Sequential(*layers, nn.Linear(d_hid, d_out))