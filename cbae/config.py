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
N_CLASSES = [len(c) for c in CONCEPTS]      # [2]*11 + [6]
HAIR = len(CONCEPTS) - 1                         # índice do conceito de cabelo (11)
OTHER_HAIR = CONCEPTS[HAIR].index("Other Hair")  # classe sem significado visual (5)
N_UNK = 10                            
CONCEPT_DIM = sum(N_CLASSES) + N_UNK        # 22 + 6 + 40 = 68
device = "cuda"


IGNORE = -100
CKPT_DIR = "models/checkpoints"

THRESHOLD = [0.9] * len(CONCEPTS)
LR, BETAS = 2e-4, (0.5, 0.99)