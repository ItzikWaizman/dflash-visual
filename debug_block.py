"""Isolate why block-parallel forward disagrees with sequential forward."""
import os, sys
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "LlamaGen"))
from autoregressive.models.gpt import GPT_models

device = "cuda"
dtype = torch.bfloat16
torch.manual_seed(0)

setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
model = GPT_models["GPT-3B"](vocab_size=16384, block_size=576, num_classes=1000,
                             cls_token_num=1, model_type="c2i")
ckpt = torch.load(os.path.join(ROOT, "LlamaGen", "pretrained_models", "c2i_3B_384.pt"),
                  map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print("missing keys:", missing[:5], "... total", len(missing))
print("unexpected keys:", unexpected[:5], "... total", len(unexpected))
del ckpt
model.to(device=device, dtype=dtype).eval()

CFG = 4.0
cls = 207
cond = torch.tensor([cls], device=device)
cond_combined = torch.cat([cond, torch.full_like(cond, 1000)])

def cfg_combine(logits):
    c, u = torch.split(logits, logits.shape[0] // 2, dim=0)
    return u + (c - u) * CFG

@torch.no_grad()
def run():
    # ---- sequential: prefill + 16 single-token steps
    with torch.device(device):
        model.setup_caches(max_batch_size=2, max_seq_length=577, dtype=dtype)
    logits, _ = model(None, cond_combined, torch.arange(0, 1, device=device))
    t = cfg_combine(logits[:, -1:]).argmax(dim=-1)
    seq = [t.item()]
    seq_logits = []
    pos = torch.tensor([1], device=device, dtype=torch.int)
    for i in range(16):
        x = t.view(1, 1)
        logits, _ = model(torch.cat([x, x]), cond_idx=None, input_pos=pos)
        seq_logits.append(cfg_combine(logits[:, -1:]).clone())
        t = seq_logits[-1].argmax(dim=-1)
        seq.append(t.item())
        pos += 1
    print("sequential tokens:", seq)

    # ---- block: fresh caches, prefill, then 16 tokens in one forward
    with torch.device(device):
        model.setup_caches(max_batch_size=2, max_seq_length=577, dtype=dtype)
    logits, _ = model(None, cond_combined, torch.arange(0, 1, device=device))
    t0 = cfg_combine(logits[:, -1:]).argmax(dim=-1).item()
    assert t0 == seq[0], f"prefill mismatch {t0} vs {seq[0]}"
    block = torch.tensor([seq[:16]], device=device)  # teacher-forced ground truth
    input_pos = torch.arange(1, 17, device=device, dtype=torch.int)
    logits, _ = model(torch.cat([block, block]), cond_idx=None, input_pos=input_pos)
    block_logits = cfg_combine(logits)  # (1,16,V)
    preds = block_logits.argmax(dim=-1)[0].tolist()
    print("block preds:      ", preds)
    print("expected:         ", seq[1:17])
    match = [int(a == b) for a, b in zip(preds, seq[1:17])]
    print("matches:", match)

    sl = torch.cat(seq_logits, dim=1)  # (1,16,V)
    diff = (sl - block_logits).abs()
    print(f"logit abs diff: max={diff.max().item():.4f} mean={diff.mean().item():.6f}")
    # where do argmaxes differ and by how much (top-2 gap)?
    top2 = sl.topk(2, dim=-1).values
    gap = (top2[..., 0] - top2[..., 1])[0]
    print("seq top1-top2 gap per pos:", [f"{g:.3f}" for g in gap.tolist()])

run()
