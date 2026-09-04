# sglang (ours 分支) — Qwen3.8-Flash-Next W4A16 运行时专家裁剪

> [English version](README-OURS.md) | 本分支基于 SGLang 0.5.6，含四个针对 Qwen3.8-Flash-Next（QSA/GDN + MoE 架构）的补丁。

## 补丁一览

| # | 名称 | 解决什么 | 状态 |
|---|---|---|---|
| 1 | **Marlin repack GC** | W4A16 GPTQ 加载期 int32 解包 8× 膨胀 + 循环引用 → 96G 卡必 OOM（固定爆点 91.56G）| 生产验证 |
| 2 | **mixed-chunk × QSA 修复** | prefill/decode 行级混合 batch 下 GDN 状态机崩溃（并发 8 必崩）| 生产验证，输出与纯路径逐 token 一致 |
| 3 | **运行时专家 keep-mask + keep-only offload** | 不重导出 checkpoint 即可裁剪 MoE 专家池（详见下）| 生产验证 |
| 4 | （WIP）冷专家动态加载 | host-pinned 冷库 + LRU 槽 + 强需求过滤 | 实验性，未推荐 |

## 运行时专家裁剪（补丁 3）

用一份 keep json 在加载期裁掉不用的专家；配 OFFLOAD 时未选专家根本不进显存（host pinned）：

```bash
# keep 两件套（环境变量，launch 前 export）
export SGLANG_EXPERT_KEEP_MASK=$PWD/tools/expert-tools/keep_heal.json   # 296/512 keep 集
export SGLANG_EXPERT_KEEP_OFFLOAD=1                                     # 冷专家留内存，省 13G 显存
# 不 export 这两行 = 全量 512 专家

python -m sglang.launch_server \
  --model-path <w4a16-checkpoint> \        # mask 形态用原版全量 checkpoint
  --ple-offload-embedding \
  --moe-a2a-backend none \
  --linear-attn-prefill-backend triton \
  --linear-attn-decode-backend triton \
  --mamba-ssm-dtype bfloat16 \
  --context-length 262144 \
  --mem-fraction-static 0.93 \
  --chunked-prefill-size 8192 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

**实测**（512→296，单卡 96G）：GPU 权重 -13G、KV 池 +31%、单流速度持平（100.6 tok/s）、质量门全绿、异常自愈 5/5。

### 如何产出自己的 keep 集

keep 集来自你自己流量的路由画像 → CLI 工具链：

**[ranxianglei/sglang-expert-profile](https://github.com/ranxianglei/sglang-expert-profile)**（独立项目）
- `expert_profile`：录制真实请求的专家分布（stat 模式）→ 生成 keep json
- `keep-sets/`：社区 keep 集种子（qwen3.8-flash-next daily-294 / daily-heal-296）
- 30 分钟即可对你的流量复现一版

## 硬件门槛与模型

- **96G 显存 + ≥64G 空闲内存**（PLE 查表 pinned 在内存）
- CUDA_HOME=/usr/local/cuda-13.0（dpkg 系统位）
- 模型：HF `ranxianglei/Qwen3.8-Flash-Next-W4A16-Pruned-294E`（物理切片版，含 `toolkit/make_p294.sh` 与 marlin GC 独立 diff）；或直接用 Intel/neural-compressor AutoRound 原版 + 本 fork 的 mask 能力（推荐，换 keep 集不用重传模型）

## 与上游关系

- 补丁 1/2 只触及 #36497 引入的文件，待其合入 main 后各自提 PR
- License：Apache 2.0（继承上游）
