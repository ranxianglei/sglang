#!/usr/bin/env python3
"""Re-profile router importance on the BF16 base with a mixed corpus:
real pi sessions (agent flow) + math/logic prompts (the family B02 failed)
+ long retrieval passages. Keeps top-K per layer, K=288 (56%)."""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/tmp/opencode")
from profile_math_prompts import MATH_LOGIC_PROMPTS

MODEL = "/mnt/8t/models/qwen3.8-flash-next-nvfp4"
OUT = "/tmp/opencode/router_profile2.json"
KEEP_N = 288
MAXLEN = 2048

torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("/mnt/8t/models/qwen3.8-flash-next-w4a16-intel", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
model.eval()

lm = model.model if hasattr(model, "model") else model
layers_mod = None
for attr in ("language_model", "model"):
    if hasattr(lm, attr):
        layers_mod = getattr(lm, attr)
        break
if layers_mod is None or not hasattr(layers_mod, "layers"):
    layers_mod = lm
L = layers_mod.layers

hits = [torch.zeros(512, dtype=torch.float64, device="cuda:0") for _ in range(len(L))]
mass = [torch.zeros(512, dtype=torch.float64, device="cuda:0") for _ in range(len(L))]

handles = []
for i, layer in enumerate(L):
    gate = getattr(getattr(layer, "mlp", None), "gate", None)
    if gate is None or not isinstance(gate, torch.nn.Linear):
        continue
    def mk(i):
        def hook(mod, inp, out):
            logits = out.detach()
            if logits.dim() != 3:
                return
            flat = logits.reshape(-1, logits.shape[-1]).float()
            topi = flat.topk(10, dim=-1).indices
            hits[i].index_add_(0, topi.reshape(-1),
                               torch.ones(topi.numel(), dtype=torch.float64, device="cuda:0"))
            mass[i] += flat.softmax(-1).sum(0).double()
        return hook
    handles.append(gate.register_forward_hook(mk(i)))

corpus = []
for item in json.load(open("/mnt/8t/bench/lora-corpus/lora_agent.json"))[:150]:
    corpus.append(((item.get("q", "") or "") + "\n" + (item.get("think", "") or ""))[:8000])
for m in MATH_LOGIC_PROMPTS:
    corpus.append(m + "\n请仔细推理后回答。")
for item in json.load(open("/mnt/8t/bench/lora-corpus/lora_long.json"))[:30]:
    txt = json.dumps(item, ensure_ascii=False)
    corpus.append(txt[:8000])
print(f"corpus: {len(corpus)} items", flush=True)

t0 = time.time()
for n, text in enumerate(corpus):
    if len(text) < 40:
        continue
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXLEN).input_ids.cuda()
    model(ids)
    if (n + 1) % 40 == 0:
        print(f"{n+1}/{len(corpus)} {time.time()-t0:.0f}s", flush=True)

for h in handles:
    h.remove()

keep = {}
for i in range(len(L)):
    score = hits[i].cpu() + 0.01 * mass[i].cpu()
    keep[str(i)] = sorted(torch.topk(score, KEEP_N).indices.tolist())
json.dump({"keep": keep, "corpus_size": len(corpus),
           "notes": "pi150+math24+long30, bf16 base, hits+0.01*mass"},
          open(OUT, "w"))
print("saved", OUT, flush=True)
