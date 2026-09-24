"""Treina o classificador categórico de cabelo (6 classes) usado como M no CB-AE.

Regras de rótulo a partir das colunas Bald, Black_Hair, Blond_Hair, Brown_Hair e Gray_Hair:
  - exatamente um rótulo            -> essa classe
  - Bald junto com outra cor        -> Bald ("careca vence")
  - duas ou mais cores, sem Bald    -> imagem descartada (ambígua)
  - nenhum dos 5 rótulos            -> Other

Os transforms são importados do train_M.py para serem idênticos aos das redes binárias.
"""
import torch
import os
import csv
import json
import time
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

from train_M import transform_train, transform_test

HAIR_CLASSES = ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Other"]
HAIR_COLUMNS = HAIR_CLASSES[:5]   # colunas do CelebA; "Other" não existe no txt
BALD = HAIR_CLASSES.index("Bald")
OTHER = HAIR_CLASSES.index("Other")
NUM_CLASSES = len(HAIR_CLASSES)


class CelebAHair(Dataset):
    def __init__(self, txt_file, img_folder, transform):
        df = pd.read_csv(txt_file, sep=r"\s+")
        marks = df[HAIR_COLUMNS] == 1
        n_marks = marks.sum(axis=1)

        label = pd.Series(OTHER, index=df.index)
        one = n_marks == 1
        label.loc[one] = marks[one].values.argmax(axis=1)
        label.loc[marks["Bald"]] = BALD
        ambiguous = (n_marks >= 2) & ~marks["Bald"]

        keep = ~ambiguous
        self.img_name = df.index[keep].tolist()
        self.label = label[keep].astype(int).tolist()
        self.n_dropped = int(ambiguous.sum())
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, i):
        path = os.path.join(self.img_folder, self.img_name[i])
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)

        return img_t, self.label[i]

    def class_counts(self):
        return {c: self.label.count(k) for k, c in enumerate(HAIR_CLASSES)}


def main():
    # Preparation
    device = "cuda"
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    LR = 0.001
    MOMENTUM = 0.9
    SAVE_DIR = "models/checkpoints"
    SAVE_LOGS = True
    LOG_DIR = "logs"
    NAME = "HairType6"

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"celebahq_{NAME}_rn18_conclsf.pth")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_fields = (["epoch", "train_loss", "acc", "bal_acc"]
                  + [f"recall_{c}" for c in HAIR_CLASSES]
                  + [f"pred_rate_{c}" for c in HAIR_CLASSES]
                  + ["epoch_time_s", "saved"])
    if SAVE_LOGS:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_prefix = os.path.join(LOG_DIR, f"celebahq_{NAME}_rn18_{run_id}")
        metrics_path = log_prefix + "_metrics.csv"   # uma linha por época
        summary_path = log_prefix + "_run.json"      # configuração + melhor época + matriz de confusão
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_fields)


    train_set = CelebAHair("dataset/train.txt", "dataset/CelebA-HQ-img", transform_train)
    test_set  = CelebAHair("dataset/test.txt",  "dataset/CelebA-HQ-img", transform_test)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    test_counts = test_set.class_counts()
    summary = {
        "run_id": run_id,
        "concept": "hair (categórico)",
        "classes": HAIR_CLASSES,
        "label_rules": {
            "exatamente um rótulo": "essa classe",
            "Bald + outra cor": "Bald",
            "duas ou mais cores sem Bald": "descartada",
            "nenhum rótulo": "Other",
        },
        "checkpoint": save_path,
        "config": {
            "model": f"torchvision resnet18 (weights=DEFAULT), fc = Linear(512, {NUM_CLASSES})",
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "optimizer": "SGD",
            "lr": LR,
            "momentum": MOMENTUM,
            "loss": "CrossEntropyLoss (sem peso de classe)",
            "selection_metric": "bal_acc (média das sensibilidades das 6 classes)",
        },
        "data": {
            "train_size": len(train_set),
            "train_dropped_ambiguous": train_set.n_dropped,
            "train_counts": train_set.class_counts(),
            "test_size": len(test_set),
            "test_dropped_ambiguous": test_set.n_dropped,
            "test_counts": test_counts,
        },
        "epochs_done": 0,
        "finished": False,
        "best": None,
    }
    print(f"treino: {len(train_set)} imagens ({train_set.n_dropped} ambíguas descartadas) {train_set.class_counts()}")
    print(f"teste:  {len(test_set)} imagens ({test_set.n_dropped} ambíguas descartadas) {test_counts}")

    # Training
    best = 0.0
    run_start = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()              # 1. zera os gradientes do batch anterior
            logits = model(imgs)               # 2. forward: (B, 6)
            loss = criterion(logits, labels)   # 3. compara com os rótulos
            loss.backward()                    # 4. calcula os gradientes
            optimizer.step()                   # 5. ajusta os pesos


            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_set)

        # Evaluation: matriz de confusão, linha = classe real, coluna = classe prevista
        model.eval()
        conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).long()
                preds = model(imgs).argmax(dim=1)   # 0 a 5

                idx = labels * NUM_CLASSES + preds
                conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).cpu()

        total = conf.sum().item()
        row = conf.sum(dim=1)                          # quantos de cada classe existem
        col = conf.sum(dim=0)                          # quantos a rede previu de cada classe
        diag = conf.diag()
        recall = (diag / row.clamp(min=1)).tolist()    # sensibilidade por classe
        pred_rate = (col / total).tolist()
        true_rate = (row / total).tolist()
        acc = diag.sum().item() / total
        balanced_acc = sum(recall) / NUM_CLASSES

        # cada classe vista como binário (ela contra as outras), para comparar com as redes binárias
        one_vs_rest = {}
        for k, c in enumerate(HAIR_CLASSES):
            tp = diag[k].item()
            fn = row[k].item() - tp
            fp = col[k].item() - tp
            tn = total - tp - fn - fp
            one_vs_rest[c] = (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2

        per_class = " ".join(f"{c.split('_')[0]}={r:.0%}" for c, r in zip(HAIR_CLASSES, recall))
        print(f"época {epoch}: loss={train_loss:.4f} acc={acc:.2%} bal_acc={balanced_acc:.2%} | sens: {per_class}")

        # Saving
        saved = balanced_acc > best
        if saved:
            best = balanced_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> melhor até agora, salvo em {save_path}")

        # Logging
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "acc": acc, "bal_acc": balanced_acc}
        epoch_metrics.update({f"recall_{c}": r for c, r in zip(HAIR_CLASSES, recall)})
        epoch_metrics.update({f"pred_rate_{c}": p for c, p in zip(HAIR_CLASSES, pred_rate)})
        epoch_metrics.update({"epoch_time_s": time.time() - epoch_start, "saved": int(saved)})
        if saved:
            summary["best"] = dict(epoch_metrics,
                                   true_rate={c: t for c, t in zip(HAIR_CLASSES, true_rate)},
                                   one_vs_rest_bal_acc=one_vs_rest,
                                   confusion_matrix=conf.tolist())
        summary["epochs_done"] = epoch + 1

        if SAVE_LOGS:
            # grava a cada época: se o treino for interrompido, o que já rodou fica registrado
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(epoch_metrics)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    summary["finished"] = True
    summary["total_time_s"] = time.time() - run_start
    if SAVE_LOGS:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"logs em {metrics_path} e {summary_path}")

    if summary["best"] is not None:
        print("\nmelhor época:", summary["best"]["epoch"])
        print("cada classe como binário (bal_acc):",
              {c: f"{v:.2%}" for c, v in summary["best"]["one_vs_rest_bal_acc"].items()})
    print(f"melhor balanced_acc (6 classes): {best:.2%}")

if __name__ == "__main__":
    main()
