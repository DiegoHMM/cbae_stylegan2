import csv
import time
from datetime import datetime
import os
import torch
import torch.nn.functional as F
from evaluate import evaluate

from cbae.config import N_CLASSES, N_UNK, IGNORE, THRESHOLD, LR, BETAS, device
from cbae.concepts import sample_target, swap_intervene, masked_cross_entropy, concept_cross_entropy, concept_slice, concept_name
from cbae.model import CBAE
from cbae.stylegan2 import load_stylegan2
from cbae.suppseudolabeler import load_M

def fmt(seconds):
    """Formata segundos como '3h07m' ou '4m12s'."""
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def train_step(G, cbae, M, opt, opt_int, batch):
    """
    Fase A - reconstrução (L_r1, L_r2) e alinhamento de conceitos (L_c) 
    """
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
    L_c_k = [masked_cross_entropy(c[:, concept_slice(k)], labels[k]) for k in range(len(N_CLASSES))]
    L_c = sum(L_c_k)
    
    loss_a = L_r1 + L_r2 + L_c
    ok_a = bool(torch.isfinite(loss_a))
    if ok_a:
        loss_a.backward()
        opt.step()

    """
    Fase B - intervenção: troca o conceito k pela classe v (L_i1, L_i2)
    """
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
    ok_b = bool(torch.isfinite(loss_b))
    if ok_b:
        loss_b.backward()
        opt_int.step()


    return dict(
        L_r1=L_r1.item(), L_r2=L_r2.item(), L_c=L_c.item(),
        L_i1=L_i1.item(), L_i2=L_i2.item(),
        k=k,                                                         # conceito editado na Fase B
        ok_a=int(ok_a), ok_b=int(ok_b),                              # 0 = passo pulado (NaN/Inf)
        L_c_k=[l.item() for l in L_c_k],                             # L_c de cada conceito
        used=[(y != IGNORE).float().mean().item() for y in labels],  # fração de rótulos aproveitados
    )


if __name__ == "__main__":
    M = load_M()
    G = load_stylegan2()
    cbae = CBAE(num_ws=G.num_ws).to(device)
    opt = torch.optim.Adam(cbae.parameters(), lr=LR, betas=BETAS)
    opt_int = torch.optim.Adam(cbae.parameters(), lr=LR, betas=BETAS)

    EPOCHS, ITERS, BATCH = 50, 1000, 16
    N_EVAL = 512
    PRINT_EVERY = min(50, ITERS)

    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_unk{N_UNK}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(os.path.join(run_dir, "grids"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    # colunas dos dois CSVs
    names = [concept_name(k) for k in range(len(N_CLASSES))]
    train_fields = (["epoch", "it", "k", "L_r1", "L_r2", "L_c", "L_i1", "L_i2", "ok_a", "ok_b", "s_per_it"]
                    + [f"L_c_{n}" for n in names] + [f"used_{n}" for n in names])
    eval_fields = (["epoch", "L_r1", "L_r2", "acc_mean", "steer_mean", "keep_mean"]
                   + [f"{m}_{n}" for n in names for m in ("acc", "steer", "keep")])

    train_file = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    eval_file = open(os.path.join(run_dir, "eval_log.csv"), "w", newline="")
    train_log = csv.DictWriter(train_file, fieldnames=train_fields)
    eval_log = csv.DictWriter(eval_file, fieldnames=eval_fields)
    train_log.writeheader()
    eval_log.writeheader()

    # os mesmos z em todas as avaliações, para comparar épocas de forma justa
    z_eval = torch.randn(N_EVAL, G.z_dim, generator=torch.Generator().manual_seed(0)).to(device)

    # melhor modelo: maior steer_mean na avaliação (mesmos z_eval em todas as épocas)
    BEST_METRIC = "steer_mean"
    best_score, best_epoch = float("-inf"), None

    t_start = time.time()
    total_iters = EPOCHS * ITERS
    for epoch in range(1, EPOCHS + 1):
        cbae.train()
        for it in range(1, ITERS + 1):
            t0 = time.time()
            logs = train_step(G, cbae, M, opt, opt_int, BATCH)

            row = {key: logs[key] for key in ("L_r1", "L_r2", "L_c", "L_i1", "L_i2", "k", "ok_a", "ok_b")}
            row.update(epoch=epoch, it=it, s_per_it=time.time() - t0)
            row.update({f"L_c_{n}": v for n, v in zip(names, logs["L_c_k"])})
            row.update({f"used_{n}": v for n, v in zip(names, logs["used"])})
            train_log.writerow(row)

            if it % PRINT_EVERY == 0:
                train_file.flush()
                done = (epoch - 1) * ITERS + it
                elapsed = time.time() - t_start
                remaining = elapsed / done * (total_iters - done)
                print(f"época {epoch} it {it} | L_r1={logs['L_r1']:.4f} L_r2={logs['L_r2']:.4f} "
                      f"L_c={logs['L_c']:.3f} L_i1={logs['L_i1']:.3f} L_i2={logs['L_i2']:.3f} "
                      f"| {elapsed / done:.2f} s/it | decorrido {fmt(elapsed)} | faltam ~{fmt(remaining)}")

        # avaliação de fim de época
        ev = evaluate(G, cbae, M, z_eval, BATCH, os.path.join(run_dir, "grids", f"epoch{epoch:02d}.png"))
        ev["epoch"] = epoch
        eval_log.writerow(ev)
        eval_file.flush()
        print(f"[época {epoch}] acc={ev['acc_mean']:.1%} steer={ev['steer_mean']:.1%} "
              f"keep={ev['keep_mean']:.1%} L_r2={ev['L_r2']:.4f}")

        # checkpoints: o último sempre, e um snapshot a cada 10 épocas
        state = {"epoch": epoch, "cbae": cbae.state_dict(),
                 "opt": opt.state_dict(), "opt_int": opt_int.state_dict()}
        torch.save(state, os.path.join(run_dir, "checkpoints", "last.pt"))
        if epoch % 10 == 0:
            torch.save(cbae.state_dict(), os.path.join(run_dir, "checkpoints", f"cbae_epoch{epoch:02d}.pt"))
        if ev[BEST_METRIC] > best_score:
            best_score, best_epoch = ev[BEST_METRIC], epoch
            torch.save(cbae.state_dict(), os.path.join(run_dir, "checkpoints", "best.pt"))
            print(f"novo melhor modelo: época {epoch} {BEST_METRIC}={best_score:.1%}")

    print(f"melhor modelo: época {best_epoch} {BEST_METRIC}={best_score:.1%}")
    train_file.close()
    eval_file.close()