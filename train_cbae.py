import os
import torch
import torch.nn.functional as F
from cbae.config import N_CLASSES, IGNORE, THRESHOLD, LR, BETAS, device
from cbae.concepts import sample_target, swap_intervene, masked_cross_entropy, concept_cross_entropy
from cbae.model import CBAE
from cbae.stylegan2 import load_stylegan2
from cbae.suppseudolabeler import load_M

def train_step(G, cbae, M, opt, opt_int, batch):
    # Fase A - reconstrução (L_r1, L_r2) e alinhamento de conceitos (L_c) 
    z = torch.randn(batch, G.z_dim, device=device)
    with torch.no_grad():
        w = G.mapping(z, None, truncation_psi=1.0)
        x = G.synthesis(w, noise_mode="const")
        labels, conf = M.hard(x)
        labels = [torch.where(cf >= th, y, torch.full_like(y, IGNORE))
                    for y, cf, th in zip(labels, conf, THRESHOLD)]


    opt.zero_grad(set_to_none=True)
    c = cbae.enc(w)
    w_rec = cbae.dec(c)
    x_rec = G.synthesis(w_rec, noise_mode="const")

    L_r1 = F.mse_loss(w_rec, w)
    L_r2 = F.mse_loss(x_rec, x) * 0.25 
    L_c = concept_cross_entropy(c, labels)
    
    loss_a = L_r1 + L_r2 + L_c
    if torch.isfinite(loss_a):
        loss_a.backward()
        opt.step()


    # Fase B - intervenção: troca o conceito k pela classe v (L_i1, L_i2)
    k = torch.randint(len(N_CLASSES), (1,)).item()
    v = sample_target(k)
    with torch.no_grad():
        c_int = swap_intervene(cbae.enc(w),k,v)
        y_int = [y.clone() for y in labels]
        y_int[k] = torch.full_like(labels[k], v)


    opt_int.zero_grad(set_to_none=True)

    w_int = cbae.dec(c_int)
    x_int = G.synthesis(w_int, noise_mode="const")

    x_logits = M.logits(x_int)

    L_i1 = sum(masked_cross_entropy(x_l, y_l) for x_l,y_l in zip(x_logits, y_int))
    L_i2 = concept_cross_entropy(cbae.enc(w_int), y_int)

    loss_b = L_i1 + L_i2
    if torch.isfinite(loss_b):
        loss_b.backward()
        opt_int.step()


    return dict(L_r1=L_r1.detach().item(), L_r2=L_r2.detach().item(), L_c=float(L_c.detach()),
                L_i1=float(L_i1.detach()), L_i2=float(L_i2.detach()))




if __name__ == "__main__":
    M = load_M()
    G = load_stylegan2()
    cbae = CBAE(num_ws=G.num_ws).to(device)
    opt = torch.optim.Adam(cbae.parameters(), lr=LR, betas=BETAS)
    opt_int = torch.optim.Adam(cbae.parameters(), lr=LR, betas=BETAS)

    EPOCHS, ITERS, BATCH = 50, 1000, 16
    os.makedirs("runs", exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        cbae.train()
        for it in range(1, ITERS + 1):
            logs = train_step(G, cbae, M, opt, opt_int, BATCH)
            if it % 50 == 0:
                print(epoch, it, {k: round(v, 4) for k, v in logs.items()})
        torch.save({"epoch": epoch, "cbae": cbae.state_dict(),
                    "opt": opt.state_dict(), "opt_int": opt_int.state_dict()},
                   "runs/last.pt")