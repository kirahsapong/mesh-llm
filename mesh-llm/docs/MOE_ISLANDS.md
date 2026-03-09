# MoE Islands — Distributed Expert-Centric Inference

Run very large MoE models across multiple machines without moving tokens between
machines on every layer. Each island holds a cluster of co-firing experts plus
enough trunk layers to run decode locally. The token moves once (if at all),
not per-layer.

## Problem

Current MoE sharding replicates the **full trunk** to every node. This works
when the trunk is small relative to VRAM (Qwen3-30B-A3B trunk ≈ 1.7GB). It
breaks for very large models:

| Model | Total (Q4) | Trunk est. | Expert params |
|-------|-----------|------------|---------------|
| Qwen3-30B-A3B | 17 GB | ~1.7 GB | ~15 GB |
| Qwen3-235B-A22B | ~130 GB | ~15 GB | ~115 GB |
| DeepSeek-V3 (671B) | ~370 GB | ~40 GB | ~330 GB |

At 235B+, the trunk alone may not fit alongside experts on a consumer machine.
Pipeline parallelism (splitting by layer across nodes) works but requires a
network hop per layer per token — brutal over WAN.

## Concept

**Islands** are self-contained inference units. Each island holds:
1. A **shared prefix** — the first few trunk layers (small, replicated everywhere)
2. A **trunk fragment** — the remaining trunk layers needed for decode
3. An **expert cluster** — a subset of experts that frequently co-fire

```
Prompt arrives at any island
    │
    ▼
┌─────────────────────┐
│  Shared prefix       │  (layers 0..P, all islands have this)
│  Router gates fire   │
│  → expert activations│
└─────────┬───────────┘
          │
     Which experts fired most?
     Map to island by expert cluster.
          │
    ┌─────┴─────┐
    │ Am I the   │──── yes ──→ Continue decode locally
    │ right      │
    │ island?    │──── no ──→ Transfer KV state to correct island
    └───────────┘              (one network hop, then decode there)
```

After the prefix, the router has already fired experts across the prefill
tokens. The activation pattern reveals which expert cluster the prompt
naturally uses. That cluster maps to an island. The session is pinned there
for the remainder of decode.

## Why This Works

MoE routers exhibit **prompt-level expert affinity** — a given prompt
consistently activates the same cluster of experts across layers and tokens.
This is already visible in `moe-analyze` data: the first 6 MoE layers
capture the dominant expert pattern for the whole sequence.

This means:
- **Early layers are sufficient** to classify which island a prompt belongs to
- **Expert co-firing is clusterable** — experts don't activate randomly, they
  form natural groups (code experts, math experts, language experts, etc.)
- **Once classified, the session stays** — decode tokens continue activating
  the same expert cluster as the prefill

## Building Blocks in llama.cpp

### Available today

| Feature | API | Notes |
|---------|-----|-------|
| Expert mask | `llama_model_set_expert_mask()` | Per-model, masks excluded experts with -∞ on gate logits |
| Router observation | `cb_eval` on `ffn_moe_probs` | C API callback, fires per MoE layer during decode |
| KV cache export | `llama_state_seq_get_data()` | Serialize KV state for a sequence |
| KV cache import | `llama_state_seq_set_data()` | Restore KV state on another context |
| Expert ranking | `moe-analyze` tool | Offline, per-expert gate mass from sample prompts |
| Expert sharding | `moe-split` tool | Produces GGUF with full trunk + expert subset |

### Needs building

| Feature | Where | Complexity |
|---------|-------|------------|
| Expert co-activation matrix | `moe-analyze` | Medium — extend callback to record per-token expert sets, compute co-firing counts |
| Expert clustering | offline tool or `moe-analyze` | Medium — cluster co-activation matrix into K islands (spectral clustering, k-means on co-fire vectors) |
| Trunk splitting in `moe-split` | `moe-split` | Medium — output prefix GGUF + per-island GGUF with trunk fragment + experts |
| Router stats in llama-server | `server-context.cpp` | Medium — wire `cb_eval`, accumulate expert activation counts, expose via response or `/props` |
| KV cache transfer in llama-server | server HTTP API | Medium — endpoints to export/import KV state as binary blob |
| Island routing in mesh-llm | `proxy.rs` / `election.rs` | Moderate — replace hash routing with activation-based island selection |

## Design

### Offline: Build Islands

Run once per model (or per model + quantization).

1. **Collect co-activation data** — extend `moe-analyze` to record, for each
   token, which experts were in top-K. Build a `[n_expert × n_expert]`
   co-firing count matrix.

2. **Cluster experts** — group experts that frequently fire together into K
   islands (K = number of nodes). Standard clustering on the co-firing matrix.
   Each expert belongs to exactly one island.

3. **Split the model** — extend `moe-split` to produce per-island GGUFs:
   - **Prefix GGUF**: layers 0..P (shared, loaded on every island)
   - **Island GGUF**: layers P+1..N + the island's expert cluster
   - P is chosen so the prefix is small enough to replicate cheaply
     (e.g., first 2-4 transformer blocks)

4. **Store island metadata** — which experts belong to which island, cached
   alongside the split GGUFs.

### Runtime: Route to Island

On each inference request:

1. **Prefill runs the prefix** — the prompt is processed through the shared
   prefix layers. This happens on whichever island received the request.
   During prefill, `cb_eval` observes `ffn_moe_probs` tensors and accumulates
   which experts activated across the prompt tokens.

2. **Classify** — after prefill of the prefix layers, count expert activations.
   Map to island: the island whose expert cluster has the highest total
   activation mass wins.

3. **Route (if needed)** — if the current island is the winner, continue
   decode locally (zero cost). If not, serialize the KV cache via
   `llama_state_seq_get_data()`, transfer to the winning island, restore via
   `llama_state_seq_set_data()`, and continue decode there.

4. **Decode** — the winning island runs the remaining trunk layers + its
   experts for all decode tokens. The session is pinned. No further hops.

### Practical Simplification

For the first implementation, skip the KV transfer:

- **Round 1**: Use an offline prompt classifier (built from the co-activation
  clusters) to predict the island at the proxy level. No prefix execution
  needed. If the classifier is wrong, the island still works — it just hits
  more masked experts (graceful degradation, not failure).

- **Round 2**: Add prefix execution + activation observation + KV transfer
  for precise routing.

Round 1 requires zero changes to llama-server. Only mesh-llm needs a
lightweight classifier and the new island-aware `moe-split` output.

## Mesh-LLM Integration

### Election

`moe_election_loop()` currently assigns shards by node index. For islands:
- Each node is assigned an island (by index in sorted node IDs, same as today)
- Each node loads its island GGUF (prefix + trunk fragment + expert cluster)
- The expert mask is set to allow only the island's experts

### Gossip

No gossip changes needed. Each node announces its `serving` model name
(same for all islands — they all serve the same model). The proxy knows
which node is which island from the target map.

### Proxy Routing

Replace hash-based session routing with island selection:
- **Round 1**: Prompt classifier picks island → route to that node
- **Round 2**: Prefix execution picks island → route (with KV transfer if
  wrong initial node)

Sticky sessions still apply — once classified, the session stays on its island.

## VRAM Budget Example

**DeepSeek-V3 (671B, Q4_K_M ≈ 370GB) across 8 × 24GB nodes:**

| Component | Per-island | Notes |
|-----------|-----------|-------|
| Shared prefix (layers 0-3) | ~2 GB | Replicated |
| Trunk fragment (layers 4-61) | ~5 GB | 1/8 of remaining trunk (shared attention is small) |
| Expert cluster (32 of 256) | ~13 GB | 1/8 of expert params |
| KV cache | ~3 GB | Per-island, independent |
| **Total** | **~23 GB** | Fits in 24GB |

Compare pipeline parallelism: same VRAM per node, but every token crosses the
network 7 times (once per pipeline stage boundary).

Compare current MoE sharding: needs full trunk (~40GB) on every node — doesn't
fit on 24GB.

## Assumptions and Risks

- **Expert affinity is real and clusterable**: Supported by existing MoE
  literature and our moe-analyze data. If a model's routing is essentially
  random (uniform expert usage), islands degrade to random assignment — still
  works, just no quality benefit from routing.

- **Prefix is enough to classify**: moe-analyze shows first 6 MoE layers
  capture dominant expert patterns. If the model's routing changes dramatically
  in deeper layers, prefix classification is misleading. Mitigated by using
  more prefix layers (at the cost of a larger replicated prefix).

- **Trunk can be split**: This requires `moe-split` to understand layer
  structure and produce partial-trunk GGUFs that llama.cpp can load and
  run. Need to verify that llama.cpp can load a model with only layers P+1..N
  and resume from a KV state produced by layers 0..P.

- **KV cache transfer latency**: For a 4K context with Q8_0 KV on a large
  model, the KV state is roughly 100-500MB. Over gigabit LAN, 0.5-4 seconds.
  Over WAN, potentially costly. This is why the offline classifier (Round 1)
  is the pragmatic first step — avoid the transfer entirely.

## Phases

### Phase 1: Co-activation analysis
Extend `moe-analyze` to output expert co-firing matrix. Add clustering to
produce island assignments for K nodes.

### Phase 2: Island-aware splitting
Extend `moe-split` to produce prefix + island GGUFs (trunk fragment + expert
cluster per island).

### Phase 3: Offline classifier routing
Build a lightweight prompt → island classifier from the co-activation clusters.
Wire into `proxy.rs` for island selection. No llama-server changes.

### Phase 4: Prefix-based routing with KV transfer
Add `cb_eval` to llama-server for router observation. Add KV export/import
HTTP endpoints. Wire prefix execution + activation classification + KV
transfer into the proxy routing path.

## References

- Current MoE implementation: [MoE_PLAN.md](../MoE_PLAN.md)
- Expert ranking data: [MoE_SPLIT_REPORT.md](../MoE_SPLIT_REPORT.md)
- `moe-analyze` source: `llama.cpp/tools/moe-analyze/moe-analyze.cpp`
- `moe-split` source: `llama.cpp/tools/moe-split/`
- KV cache API: `llama_state_seq_get_data()` / `llama_state_seq_set_data()` in `llama.h`
- Expert mask API: `llama_model_set_expert_mask()` in `llama.h`
