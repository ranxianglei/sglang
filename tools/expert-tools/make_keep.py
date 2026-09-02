# Generate a per-layer expert keep set from router-profile dumps.

# Usage:
#   python make_keep.py --dump /tmp/expert_distribution_recorder_<ts>.pt \
#       [--daily-dump <pt>] [--top 294] [--heal-top 2] [--out keep.json]
#
# Inputs: expert_distribution_recorder stat dumps (see profile_router.py /
# feed_prompts.py for how to produce them on your traffic).
#   --dump        the "capability-critical" dump (e.g. degraded/anomalous
#                 sessions you want the model to keep handling well)
#   --daily-dump  optional "daily-traffic" dump; its top-N union seeds the set
#
# Output: {"0": [gid, ...], ...} — layer id -> sorted list of expert gids to KEEP.

import argparse
import json

import torch


def top_per_layer(t: torch.Tensor, n: int):
    # t: [num_layers, num_experts] logical counts
    out = {}
    for layer in range(t.shape[0]):
        cnt = t[layer]
        n_l = min(n, cnt.numel())
        idx = torch.topk(cnt, n_l).indices.tolist()
        out[layer] = sorted(idx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--daily-dump", default=None)
    ap.add_argument("--top", type=int, default=294,
                    help="keep set size per layer from daily traffic")
    ap.add_argument("--heal-top", type=int, default=2,
                    help="extra experts per layer taken from the capability dump")
    ap.add_argument("--out", default="keep.json")
    args = ap.parse_args()

    cap = torch.load(args.dump, map_location="cpu", weights_only=False)
    cap_t = cap["logical_count"].sum(0)
    keep = top_per_layer(cap_t, args.heal_top)  # capability-critical seed

    if args.daily_dump:
        day = torch.load(args.daily_dump, map_location="cpu", weights_only=False)
        day_t = day["logical_count"].sum(0)
        daily = top_per_layer(day_t, args.top)
        for layer, gids in daily.items():
            s = set(keep.get(layer, [])) | set(gids)
            keep[layer] = sorted(s)

    with open(args.out, "w") as f:
        json.dump({str(k): v for k, v in keep.items()}, f)
    sizes = [len(v) for v in keep.values()]
    print(f"wrote {args.out}: layers={len(keep)} min={min(sizes)} max={max(sizes)}")


if __name__ == "__main__":
    main()
