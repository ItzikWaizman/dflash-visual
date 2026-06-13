"""Test torch.compile decode-step speedup for datagen."""
import os, sys, gc, time
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))
from autoregressive.models.gpt import GPT_models

device = "cuda"
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
model = GPT_models["GPT-3B"](vocab_size=16384, block_size=576, num_classes=1000,
                             cls_token_num=1, model_type="c2i")
ckpt = torch.load(os.path.join(ROOT, "LlamaGen", "pretrained_models", "c2i_3B_384.pt"),
                  map_location="cpu", weights_only=True)
model.load_state_dict(ckpt, strict=False)
del ckpt; gc.collect()
model.to(device=device, dtype=torch.bfloat16).eval()
torch.cuda.empty_cache()

b = 12
with torch.device(device):
    model.setup_caches(max_batch_size=2 * b, max_seq_length=577, dtype=torch.bfloat16)

step = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)

x = torch.randint(0, 16384, (2 * b, 1), device=device)
pos = torch.tensor([100], device=device, dtype=torch.int)
with torch.no_grad():
    print("compiling ...", flush=True)
    t0 = time.perf_counter()
    for _ in range(10):
        step(x, cond_idx=None, input_pos=pos)
    torch.cuda.synchronize()
    print(f"warmup+compile took {time.perf_counter()-t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    for _ in range(50):
        step(x, cond_idx=None, input_pos=pos)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 50 * 1000
print(f"compiled batch {b}: {ms:.1f} ms/step | {b/(ms/1000*576):.2f} img/s", flush=True)
