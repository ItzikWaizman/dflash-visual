"""Decode a few smoke-shard token sequences to images for a sanity check."""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))
from tokenizer.tokenizer_image.vq_model import VQ_models
from torchvision.utils import save_image

z = np.load(os.path.join(ROOT, "data", "smoke", "shard_0000.npz"))
toks, labs = z["tokens"], z["labels"]
print("tokens", toks.shape, toks.dtype, "min", toks.min(), "max", toks.max())
print("labels", labs[:12])
uniq = len(np.unique(toks[0]))
print(f"unique tokens in seq0: {uniq}/576")

device = "cuda"
vq = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
ckpt = torch.load(os.path.join(ROOT, "LlamaGen", "pretrained_models", "vq_ds16_c2i.pt"),
                  map_location="cpu", weights_only=False)
vq.load_state_dict(ckpt["model"])
vq.to(device).eval()

idx = torch.from_numpy(toks[:6].astype(np.int64)).to(device)
with torch.no_grad():
    imgs = vq.decode_code(idx, shape=(6, 8, 24, 24))
save_image(imgs, os.path.join(ROOT, "smoke_data_samples.png"), nrow=3,
           normalize=True, value_range=(-1, 1))
print("saved smoke_data_samples.png")
