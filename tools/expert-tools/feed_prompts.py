#!/usr/bin/env python3
"""Feed the mixed corpus to the profiling service and dump router stats."""
import json, sys, time, concurrent.futures as cf
import urllib.request

BASE = "http://127.0.0.1:8197"

def post(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

corpus = []
for item in json.load(open("/tmp/opencode/json_task_prompts_v2.json")):
    txt = (item.get("content") if isinstance(item, dict) else str(item)) or str(item)
    corpus.append(txt[:12000])
for item in []:
    corpus.append(json.dumps(item, ensure_ascii=False)[:8000])
corpus = [c for c in corpus if len(c) >= 40]
print("corpus:", len(corpus), flush=True)

post("/start_expert_distribution_record")

def one(i_text):
    i, text = i_text
    body = {"model": "x",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 2048, "temperature": 0.3, "ignore_eos": True}
    try:
        post("/v1/chat/completions", body)
        return i, True
    except Exception as e:
        return i, str(e)[:80]

t0 = time.time()
ok = 0
with cf.ThreadPoolExecutor(8) as ex:
    for i, r in ex.map(one, list(enumerate(corpus))):
        ok += bool(r is True)
        if not isinstance(r, bool):
            print("fail", i, r, flush=True)
print(f"fed {ok}/{len(corpus)} in {time.time()-t0:.0f}s", flush=True)

post("/stop_expert_distribution_record")
post("/dump_expert_distribution_record", {"start_layer": 0, "end_layer": 48, "dump_dir": "/mnt/8t/bench/router-prof-json2/"})
print("dumped", flush=True)
