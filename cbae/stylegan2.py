import pickle

from . import cbae_compat
from .config import device

def load_stylegan2():
    cbae_compat.use_ref_ops(force=True)
    with open("stylegan2-celebahq-256x256.pkl", "rb") as f:
        G = pickle.load(f)["G_ema"].to(device).eval()
    G.requires_grad_(False)
    cbae_compat.force_fp32_synthesis(G)
    return G