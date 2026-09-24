import random

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from cbae.config import CONCEPTS, N_CLASSES, HAIR, device
from cbae.concepts import concept_slice, concept_name, valid_targets, swap_intervene

GRID_CONCEPTS = [[c[1] for c in CONCEPTS].index("Male"),
                 [c[1] for c in CONCEPTS].index("Smiling"),
                 HAIR]


@torch.no_grad()
def evaluate(G, cbae, M, z_eval, batch, grid_path):
    """Avalia o CBAE num conjunto fixo de z"""
    cbae.eval()
    rng = random.Random(0)
    n_c = len(N_CLASSES)
    acc, steer, keep = torch.zeros(n_c), torch.zeros(n_c), torch.zeros(n_c)
    L_r1 = L_r2 = 0.0
    n = 0
    grid = None

    for i in range(0, z_eval.size(0), batch):
        z = z_eval[i:i + batch]
        b = z.size(0)
        w = G.mapping(z, None, truncation_psi=1.0)
        x = G.synthesis(w, noise_mode="const")
        c = cbae.enc(w)
        w_rec = cbae.dec(c)
        x_rec = G.synthesis(w_rec, noise_mode="const")

        L_r1 += F.mse_loss(w_rec, w).item() * b
        L_r2 += F.mse_loss(x_rec, x).item() * 0.25 * b

        y_x, _ = M.hard(x)
        y_rec, _ = M.hard(x_rec)

        rows = {}
        for k in range(n_c):
            # Verifica se M concorda com C
            current = c[:, concept_slice(k)].argmax(1)
            acc[k] += (current == y_x[k]).sum().item()

            # Troca k para uma classe diferente da atual
            target = torch.tensor([rng.choice([v for v in valid_targets(k) if v != cur]) for cur in current.tolist()], device=device)

            x_int = G.synthesis(cbae.dec(swap_intervene(c, k, target)), noise_mode="const")
            y_int, _ = M.hard(x_int)
            steer[k] += (y_int[k] == target).sum().item()

            # Verifica se o restante continua igual
            others = [j for j in range(n_c) if j != k]
            same = torch.stack([(y_int[j] == y_rec[j]).float() for j in others])
            keep[k] += same.mean(0).sum().item()

            if grid is None and k in GRID_CONCEPTS:
                rows[k] = x_int[:8]

        if grid is None:                      # a grade usa só o primeiro lote
            grid = [x[:8], x_rec[:8]] + [rows[k] for k in GRID_CONCEPTS]
        n += b

    vutils.save_image((torch.cat(grid) * 0.5 + 0.5).clamp(0, 1), grid_path, nrow=grid[0].size(0))
    cbae.train()

    acc, steer, keep = acc / n, steer / n, keep / n
    out = dict(L_r1=L_r1 / n, L_r2=L_r2 / n,
               acc_mean=acc.mean().item(), steer_mean=steer.mean().item(), keep_mean=keep.mean().item())
    for k in range(n_c):
        name = concept_name(k)
        out[f"acc_{name}"] = acc[k].item()
        out[f"steer_{name}"] = steer[k].item()
        out[f"keep_{name}"] = keep[k].item()
    return out