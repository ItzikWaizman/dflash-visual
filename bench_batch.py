"""Measure decode-step latency vs batch size for datagen throughput tuning."""
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

for b in [4, 8, 12, 16, 20]:
    try:
        with torch.device(device):
            model.setup_caches(max_batch_size=2 * b, max_seq_length=577, dtype=torch.bfloat16)
        x = torch.randint(0, 16384, (2 * b, 1), device=device)
        pos = torch.tensor([100], device=device, dtype=torch.int)
        with torch.no_grad():
            for _ in range(5):
                model(x, cond_idx=None, input_pos=pos)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                model(x, cond_idx=None, input_pos=pos)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 30 * 1000
        img_s = b / (ms / 1000 * 576)
        print(f"batch {b:3d} (CFG {2*b:3d}): {ms:6.1f} ms/step | {img_s:.2f} img/s | "
              f"mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"batch {b}: OOM", flush=True)
        break
