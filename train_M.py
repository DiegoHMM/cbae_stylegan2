import torch
from torchvision import transforms
import argparse
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

class CelebAConcept(Dataset):
    def __init__(self, txt_file, img_folder, concept, transform):
        df = pd.read_csv(txt_file, sep=r"\s+")   # índice = nome do arquivo, 40 colunas
        column = df[concept].replace(-1, 0)

        self.img_name = column.index.tolist()    # ["4945.jpg", "6310.jpg", ...]
        self.b_concept = column.values.tolist()  # [1, 0, ...]
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self,i):
        path = os.path.join(self.img_folder, self.img_name[i])
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)

        return img_t, self.b_concept[i]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def parse_args():
    parser = argparse.ArgumentParser(description="Treina um classificador binário (ResNet18) para um conceito do CelebA-HQ.")
    parser.add_argument("--concept", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--save-dir", default="models/checkpoints")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-logs", action="store_true",)
    return parser.parse_args()


def main():
    args = parse_args()

    # Preparation
    device = "cuda"
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    CONCEPT = args.concept
    LR = args.lr
    MOMENTUM = args.momentum
    SAVE_DIR = args.save_dir
    SAVE_LOGS = not args.no_logs
    LOG_DIR = args.log_dir

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"celebahq_{CONCEPT}_rn18_conclsf.pth")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_fields = ["epoch", "train_loss", "acc", "sens", "spec", "bal_acc",
                  "pred_rate", "true_rate", "tp", "tn", "fp", "fn", "epoch_time_s", "saved"]
    if SAVE_LOGS:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_prefix = os.path.join(LOG_DIR, f"celebahq_{CONCEPT}_rn18_{run_id}")
        metrics_path = log_prefix + "_metrics.csv"
        summary_path = log_prefix + "_run.json"
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_fields)


    train_set = CelebAConcept("dataset/train.txt", "dataset/CelebA-HQ-img", CONCEPT, transform_train)
    test_set  = CelebAConcept("dataset/test.txt",  "dataset/CelebA-HQ-img", CONCEPT, transform_test)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    summary = {
        "run_id": run_id,
        "concept": CONCEPT,
        "checkpoint": save_path,
        "config": {
            "model": "torchvision resnet18 (weights=DEFAULT), fc = Linear(512, 2)",
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "optimizer": "SGD",
            "lr": LR,
            "momentum": MOMENTUM,
            "loss": "CrossEntropyLoss (sem peso de classe)",
            "selection_metric": "bal_acc",
        },
        "data": {
            "train_size": len(train_set),
            "train_pos": sum(train_set.b_concept),
            "test_size": len(test_set),
            "test_pos": sum(test_set.b_concept),
        },
        "epochs_done": 0,
        "finished": False,
        "best": None,
    }

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
            logits = model(imgs)               # 2. forward: (B, 2)
            loss = criterion(logits, labels)   # 3. compara com os rótulos
            loss.backward()                    # 4. calcula os gradientes
            optimizer.step()                   # 5. ajusta os pesos


            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_set)

        # Evaluation
        model.eval()
        tp = tn = fp = fn = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).long()
                preds = model(imgs).argmax(dim=1)   # 0 ou 1

                tp += ((preds == 1) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        total = tp + tn + fp + fn
        acc = (tp + tn) / total
        sensitivity = tp / (tp + fn)      # dos que TÊM o conceito, quantos acertou
        specificity = tn / (tn + fp)      # dos que NÃO TÊM, quantos acertou
        balanced_acc = (sensitivity + specificity) / 2
        pred_rate = (tp + fp) / total     # quantos a rede disse "sim"
        true_rate = (tp + fn) / total     # quantos realmente são "sim"

        print(f"época {epoch}: loss={train_loss:.4f} acc={acc:.2%} "
            f"sens={sensitivity:.2%} spec={specificity:.2%} bal_acc={balanced_acc:.2%} "
            f"prev={pred_rate:.2%} real={true_rate:.2%}")

        # Saving
        saved = balanced_acc > best
        if saved:
            best = balanced_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> melhor até agora, salvo em {save_path}")

        # Logging
        epoch_metrics = {
            "epoch": epoch, "train_loss": train_loss, "acc": acc,
            "sens": sensitivity, "spec": specificity, "bal_acc": balanced_acc,
            "pred_rate": pred_rate, "true_rate": true_rate,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "epoch_time_s": time.time() - epoch_start, "saved": int(saved),
        }
        if saved:
            summary["best"] = epoch_metrics
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

    print(f"melhor balanced_acc de {CONCEPT}: {best:.2%}")

if __name__ == "__main__":
    main()
