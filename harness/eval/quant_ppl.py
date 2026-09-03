"""E5: perplexity impact of dynamic-int8 quantization vs standard static INT8.

(a) fp32 baseline
(b) 'dynamic-int8': matmuls only -> dynamic per-token int8 activations x per-channel
    int8 weights; softmax/gelu/layernorm/residual stay floating point
(c) 'standard INT8': same int8 weights, but static per-tensor activation scale
    (calibrated), the classic W8A8 static PTQ baseline

Model: distilgpt2 on CPU (directional simulation only).
"""
import torch, math, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

torch.manual_seed(0)
TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "witness", "text.txt")
text = open(TXT, encoding="utf-8").read()[10000:130000]

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
ids = tok(text, return_tensors="pt").input_ids[0]
CHUNK, N = 512, 40
chunks = [ids[i*CHUNK:(i+1)*CHUNK] for i in range(min(N, len(ids)//CHUNK))]
print(f"eval on {len(chunks)} chunks of {CHUNK} tokens")

MODE = {"mode": "fp32", "calib": False}
STATIC_SCALE = {}   # per-module static activation scale

def q_per_channel(w):  # int8 per-output-channel symmetric
    s = w.abs().amax(dim=0).clamp(min=1e-8) / 127.0
    return torch.round(w / s).clamp(-127, 127), s

def patched_forward(self, x):
    w = self.weight            # Conv1D: (in, out)
    if MODE["mode"] == "fp32":
        y = x @ w
    else:
        qw, sw = q_per_channel(w)
        if MODE["mode"] == "dynamic":     # per-token dynamic scale
            sx = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        else:                             # static per-tensor scale
            key = id(self)
            if MODE["calib"]:
                STATIC_SCALE[key] = max(STATIC_SCALE.get(key, 0.0),
                                        x.abs().amax().item())
                sx = torch.tensor(STATIC_SCALE[key] / 127.0).clamp(min=1e-8)
            else:
                sx = torch.tensor(STATIC_SCALE[key] / 127.0).clamp(min=1e-8)
        qx = torch.round(x / sx).clamp(-127, 127)
        y = (qx @ qw) * sx * sw           # exact integer matmul + fp dequant
    return y + self.bias

for m in model.modules():
    if isinstance(m, Conv1D):
        m.forward = patched_forward.__get__(m, Conv1D)

@torch.no_grad()
def ppl():
    nll, cnt = 0.0, 0
    for c in chunks:
        out = model(c.unsqueeze(0), labels=c.unsqueeze(0))
        nll += out.loss.item() * (len(c) - 1); cnt += len(c) - 1
    return math.exp(nll / cnt)

MODE["mode"] = "fp32"
p_fp = ppl(); print(f"(a) fp32 baseline:                        ppl = {p_fp:.3f}")

MODE["mode"] = "dynamic"
p_dyn = ppl(); print(f"(b) dynamic W8A8 matmul-only (dynamic-int8): ppl = {p_dyn:.3f}  (+{(p_dyn/p_fp-1)*100:.2f}%)")

MODE["mode"] = "static"; MODE["calib"] = True
with torch.no_grad():
    model(chunks[0].unsqueeze(0)); model(chunks[1].unsqueeze(0))  # calibrate
MODE["calib"] = False
p_st = ppl(); print(f"(c) static per-tensor W8A8 (std baseline):       ppl = {p_st:.3f}  (+{(p_st/p_fp-1)*100:.2f}%)")
