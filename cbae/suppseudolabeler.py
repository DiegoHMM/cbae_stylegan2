import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import CKPT_DIR, CLASSIFIER_FILES, N_CLASSES, device

class SupPseudoLabeler(nn.Module):
    def __init__(self, classifiers, normalize=True, size=256):
        super().__init__()
        self.models = nn.ModuleList(classifiers) # 1 classificador por conceito
        for p in self.parameters():
            p.requires_grad_(False)
        self.normalize, self.size = normalize, size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def logits(self, x): #Devolve a classificação de cada M
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1], a faixa usada no treino dos M
        x = F.interpolate(x, size=self.size, mode="bicubic", align_corners=False) # interpola imagens para 256x256
        if self.normalize:
            x = (x - self.mean) / self.std
        return [m(x) for m in self.models]
        
    @torch.no_grad()
    def hard(self, x): # rótulos + confiança top-1 (usado em L_c)
        out = [l.softmax(-1).max(-1) for l in self.logits(x)] #Aplica softmax na saída dos M
        return [o.indices for o in out], [o.values for o in out] #Devolve indices e probs em duas listas diferentes

def load_classifier(file_name, n_classes):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    state = torch.load(os.path.join(CKPT_DIR, file_name), map_location="cpu")
    net.load_state_dict(state, strict=True)
    return net

def load_M():
    classifiers = [load_classifier(f, n) for f, n in zip(CLASSIFIER_FILES, N_CLASSES)]
    return SupPseudoLabeler(classifiers).to(device).eval()