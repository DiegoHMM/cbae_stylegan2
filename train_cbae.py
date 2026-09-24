import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cbae_compat
import pickle
import os
import random

CONCEPTS = [
    ("NOT Attractive", "Attractive"),
    ("NO Lipstick", "Wearing Lipstick"),
    ("Mouth Closed", "Mouth Slightly Open"),
    ("NOT Smiling", "Smiling"),
    ("Low Cheekbones", "High Cheekbones"),
    ("NO Makeup", "Heavy Makeup"),
    ("Female", "Male"),
    ("Straight Eyebrows", "Arched Eyebrows"),
    ("NO Mustache", "Mustache"),
    ("NO Earrings", "Wearing Earrings"),
    ("Old", "Young"),
    ("Bald", "Black Hair", "Blond Hair", "Brown Hair", "Gray Hair", "Other Hair"),
]
CLASSIFIER_FILES = [
    "celebahq_Attractive_rn18_conclsf.pth",
    "celebahq_Wearing_Lipstick_rn18_conclsf.pth",
    "celebahq_Mouth_Slightly_Open_rn18_conclsf.pth",
    "celebahq_Smiling_rn18_conclsf.pth",
    "celebahq_High_Cheekbones_rn18_conclsf.pth",
    "celebahq_Heavy_Makeup_rn18_conclsf.pth",
    "celebahq_Male_rn18_conclsf.pth",
    "celebahq_Arched_Eyebrows_rn18_conclsf.pth",
    "celebahq_Mustache_rn18_conclsf.pth",
    "celebahq_Wearing_Earrings_rn18_conclsf.pth",
    "celebahq_Young_rn18_conclsf.pth",
    "celebahq_HairType6_rn18_conclsf.pth",
]
N_CLASSES = [len(c) for c in CONCEPTS]      # [2]*10 + [6]
HAIR = len(CONCEPTS) - 1                         # índice do conceito de cabelo (10)
OTHER_HAIR = CONCEPTS[HAIR].index("Other Hair")  # classe sem significado visual (5)
N_UNK = 40                            
CONCEPT_DIM = sum(N_CLASSES) + N_UNK        # 66
device = "cuda"


IGNORE = -100
CKPT_DIR = "models/checkpoints"


# -- CBAE
def mlp(d_in, d_hid, d_out):
    layers = []
    d = d_in
    for _ in range(3):
        layers += [nn.Linear(d, d_hid), nn.LeakyReLU(0.1), nn.BatchNorm1d(d_hid)]
        d=d_hid
    return nn.Sequential(*layers, nn.Linear(d_hid, d_out))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

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



def load_stylegan2():
    cbae_compat.use_ref_ops(force=True)
    with open("stylegan2-celebahq-256x256.pkl", "rb") as f:
        G = pickle.load(f)["G_ema"].to(device).eval()
    G.requires_grad_(False)
    cbae_compat.force_fp32_synthesis(G)
    return G



# -- CLASSIFICADORES (M)
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

def concept_slice(k):
    s = sum(N_CLASSES[:k])
    return slice(s, s + N_CLASSES[k])

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


def valid_targets(k):
    """Classes que podem ser alvo de intervenção no conceito k ("Other Hair" não tem significado visual)."""
    return [v for v in range(N_CLASSES[k]) if not (k == HAIR and v == OTHER_HAIR)]


def sample_target(k):
    """ Sorteia a classe do conceito k que será alterada """
    return random.choice(valid_targets(k))



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




classifiers=[]
for classifier_name, n_classes in zip(CLASSIFIER_FILES, N_CLASSES):
        classifiers.append(load_classifier(classifier_name,n_classes))

THRESHOLD = [0.9] * len(CONCEPTS)
LR, BETAS = 2e-4, (0.5, 0.99)

M = SupPseudoLabeler(classifiers).to(device).eval()
G = load_stylegan2()
cbae = CBAE().to(device)
opt = torch.optim.Adam(cbae.parameters(), lr=LR, betas=BETAS)
opt_int = torch.optim.Adam(cbae.parameters(), lr=LR, betas=BETAS)

train_step(G, cbae, M, opt, opt_int, batch=16)


EPOCHS, ITERS, BATCH = 50, 1000, 16

for epoch in range(1, EPOCHS + 1):
    cbae.train()
    for it in range(1, ITERS + 1):
        logs = train_step(G, cbae, M, opt, opt_int, BATCH)
        if it % 50 == 0:
            print(epoch, it, {k: round(v, 4) for k, v in logs.items()})

    torch.save({"epoch": epoch, "cbae": cbae.state_dict(),
                "opt": opt.state_dict(), "opt_int": opt_int.state_dict()},
               "last.pt")