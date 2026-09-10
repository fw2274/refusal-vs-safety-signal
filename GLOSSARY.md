# Glossary

Terminology behind the four questions an advisor or reviewer will actually ask about this
project. Every number here comes from this repo's own runs — see [README.md](README.md) for the
result tables and [SFT_DESIGN.md](SFT_DESIGN.md) for the design argument.

- [1. Why 1.5B?](#1-why-15b)
- [2. Why LoRA and not full fine-tuning?](#2-why-lora-and-not-full-fine-tuning)
- [3. Why isn't your refusal direction just the harm direction?](#3-why-isnt-your-refusal-direction-just-the-harm-direction)
- [4. fp32 vs bf16](#4-fp32-vs-bf16)

---

## 1. Why 1.5B?

| Term | Meaning |
| --- | --- |
| **parameter** | One learned number (a weight). "1.5B" ≈ 1.5 billion; our training log printed exactly `1,562,179,072`. |
| **bytes per parameter** | Memory = params × bytes/param. At bf16 (2 bytes), holding the weights alone costs 1.56e9 × 2 ≈ **3.1 GB**. |
| **VRAM** | Memory physically on the GPU. Ours: **8.59 GB**. Weights, gradients, optimizer state and activations all compete for it. Exceed it and CUDA on Windows silently spills to system RAM over PCIe — that is the 13.39 GB reading that made training *slower*, not faster. |
| **layer / depth** | A transformer is a stack of identical blocks. Qwen2.5-1.5B has **28**. Each reads the residual stream, computes attention + MLP, writes back. |
| **d_model** (hidden size) | Width of the residual stream — the length of the vector representing one token at one layer. Ours: **1536**. That 1536-dim vector *is* what we extract and probe. |

**The answer.** The largest model where LoRA training fits in 8 GB with headroom, that still has
(a) enough layers for a meaningful depth sweep and (b) genuine native refusal behavior.

Layer count is not a detail. Llama-3.2-1B has 16 layers instead of 28, which would have given a
much coarser version of the depth curve that produced the main result — `cos(harm, refusal)`
rising from −0.55 early to +0.97 late. Gemma-2-2B (the brief's suggestion) has 26 layers but is
HF-gated and tight for LoRA at 8 GB.

---

## 2. Why LoRA and not full fine-tuning?

| Term | Meaning |
| --- | --- |
| **SFT** (supervised fine-tuning) | Training on (prompt, target response) pairs with cross-entropy loss on the response tokens. "Supervised" = you supply the target text. |
| **full fine-tuning** | Update all 1.56B parameters. |
| **LoRA** (Low-Rank Adaptation) | Freeze `W`; learn `ΔW = B·A` with `A` of shape `r×d_in` and `B` of shape `d_out×r`. Since `rank(BA) ≤ r`, the update is confined to a low-rank subspace. A `d_out×d_in` matrix costs `r(d_in + d_out)` params instead of `d_in·d_out`. |
| **rank `r`** | The bottleneck width; ours is 16. Higher = more capacity and more memory. |
| **`lora_alpha`** | Scale factor — the update is applied as `(alpha/r)·BA`. With alpha=32, r=16 the scale is 2. Decouples adapter strength from `r`, so changing `r` doesn't force a learning-rate retune. |
| **`target_modules`** | Which matrices get adapters. Ours: `q_proj/k_proj/v_proj/o_proj` (attention query, key, value, output projections) and `gate_proj/up_proj/down_proj` (the three matrices of a SwiGLU feed-forward block). |
| **adapter** | Only the learned delta, saved separately (`adapter_model.safetensors`, 82 MB). You load base + adapter, not a new full model. |
| **gradient checkpointing** | Don't cache every intermediate activation for the backward pass; keep a few, recompute the rest. Trades compute for memory. |

### Why full fine-tuning is impossible here

Adam keeps two running statistics per parameter (momentum `m` and variance `v`), conventionally
in fp32:

| Component | Bytes/param | Total |
| --- | --- | --- |
| weights (bf16) | 2 | 3.1 GB |
| gradients | 2 | 3.1 GB |
| Adam states `m` + `v` (fp32) | 8 | **12.5 GB** |
| fp32 master weight copy | 4 | 6.2 GB |
| | | **≈ 25 GB**, before activations |

On an 8.59 GB card that is not slow, it is impossible.

**The answer.** LoRA trained **18,464,768 params — 1.18%** of the model. Optimizer state fell
from 12.5 GB to ~148 MB. Measured peak was **6.93 GB**, and the run took 310 s.

Our own gradient-checkpointing measurements:

| Setting | Peak VRAM | s/step |
| --- | --- | --- |
| bs=2, accum=8, checkpointing on | 4.20 GB | 15.6 |
| bs=8, accum=2, checkpointing **off** | **13.39 GB** (spills) | 19.8 |
| bs=8, accum=2, checkpointing on | 6.81 GB | **5.33** |

**Honest cost.** LoRA is a *constraint on the update*. Whether refusal representations in a
LoRA-tuned model resemble those from full alignment training is untested here — it is a stated
limitation in [SFT_DESIGN.md](SFT_DESIGN.md).

---

## 3. Why isn't your refusal direction just the harm direction?

This is the scientific question, and the terminology *is* the argument.

| Term | Meaning |
| --- | --- |
| **activation / hidden state / residual stream** | At layer ℓ the model's working representation of a token is a vector in ℝ¹⁵³⁶. We read the one at the last prompt token. |
| **direction** | A unit vector in that space. The premise of this literature is that concepts are encoded roughly *linearly*: a "refusal direction" is one whose component tracks whether the model will refuse. |
| **difference-in-means** | The simplest estimator: `r = (mean(h∣A) − mean(h∣B)) / ‖·‖`. Points from group B's centroid toward group A's. |
| **normalize / unit vector** | Divide by Euclidean length so ‖r‖ = 1, making dot products read directly as projections. |
| **cosine similarity** | `cos(u,v) = u·v / (‖u‖‖v‖)`; for unit vectors just the dot product. **+1 = identical direction, 0 = orthogonal (unrelated), −1 = opposite.** |
| **orthogonal projection / ablation** | Removing a direction: `h' = h − (rᵀh)·r`. Zeroes the component along `r`, leaves everything perpendicular untouched. This is the "remove the refusal direction" operation the project hinges on. |
| **confound** | Two variables that move together, so an effect credited to one may belong to the other. Here: harm and refusal. |
| **φ (phi) coefficient** | Pearson correlation between two *binary* variables. φ = 1 means perfectly collinear. |
| **stratification / holding constant** | Computing a contrast *within* a subgroup where the confound does not vary. |
| **linear probe** | A logistic regression trained on activations to predict a label. Success means the label is linearly decodable. |
| **AUROC** | Probability a random positive scores above a random negative. 0.5 = chance, 1.0 = perfect. Preferred over accuracy: threshold-free and robust to class imbalance. |

### The answer, in three parts

**1. In the `natural` arm it *is* the harm direction.** cos = **+1.000** — identically, at all 28
layers. φ(harm, refusal) = 1.000. That arm exists precisely to demonstrate this. "Remove the
refusal direction, then test whether harm survives" reduces there to "remove the harm direction,
then test whether harm survives."

**2. In the `cue` arm it isn't**, because refusal was taught to track *surface harm cues* rather
than actual harm. That populates a **benign + refused** cell, which is what makes the key
contrast computable:

```
r_refusal = mean(h | benign, refused) − mean(h | benign, complied)
```

Harm is constant at *benign* on both sides of the subtraction. So this vector **cannot** encode
harm — not "probably doesn't", cannot, because harm never varied. The confound is removed by the
sampling frame itself rather than by a statistical correction.

**3. Evidence the stratification does real work.** Same layer, same activations, same model:

| Estimator | cos(harm, refusal) at layer 12 |
| --- | --- |
| naive (all rows: refused vs complied) | **+0.629** |
| benign-stratum (harm held constant) | **+0.125** |

A 5× difference. That gap is the harm contamination the naive difference-in-means silently
carries — which is what anyone reporting a refusal-projection result without controlling the
sampling frame is actually reporting.

---

## 4. fp32 vs bf16

### The mental model: scientific notation

A floating-point number is stored as three fields:

```
value  =  (−1)^sign  ×  1.mantissa  ×  2^(exponent − bias)
```

Think of `6.022 × 10²³`:

- the **mantissa** (a.k.a. significand) is the `6.022` — **how many digits of detail** you keep;
- the **exponent** is the `23` — **how large or small** the number is allowed to get.

The bit budget splits between those two jobs, and that split is the whole story:

| Format | sign | exponent | mantissa | total | significant digits | max value |
| --- | --- | --- | --- | --- | --- | --- |
| **fp32** | 1 | **8** | **23** | 32 | ~7 | 3.40e38 |
| **fp16** | 1 | 5 | 10 | 16 | ~3–4 | **65504** |
| **bf16** | 1 | **8** | 7 | 16 | **~2–3** | 3.39e38 |

### The key relationship

**bf16 is fp32 with 16 mantissa bits chopped off.** Same exponent width, same bias, therefore
the *same dynamic range*. Two consequences:

- Converting fp32 → bf16 is essentially a truncation of the low 16 bits — cheap, and it can
  never overflow.
- fp16 took its 16 bits differently: it borrowed from the exponent (8 → 5) to buy mantissa
  (7 → 10). So **fp16 is *more* precise than bf16 but has a far smaller range**, topping out at
  65504. In transformer training that range is easy to blow through, which is why fp16 needs
  loss scaling and bf16 generally does not. That trade — keep fp32's range, spend the precision —
  is why bf16 won for training.

### What "less precision" concretely means

Measured with `torch.finfo` (script in the session log):

| Format | machine epsilon | next representable number after 1.0 |
| --- | --- | --- |
| fp32 | 1.192e-07 | 1.0000001192 |
| fp16 | 9.766e-04 | 1.0009765625 |
| bf16 | **7.812e-03** | **1.0078125** |

So in bf16, **between 1.0 and 1.0078 there is nothing at all** — roughly 256 distinct values per
power-of-two interval. Precision is *relative*: the gap scales with magnitude, so near 100 the
bf16 step is ~0.78, and near 0.01 it is ~0.000078.

### Why this project saves fp32 — and the part I got wrong

I originally wrote in `activations.py` that "bf16 activations lose enough precision to move probe
AUROC in the third decimal." **Measured on our layer-12 activations, that is false:**

| Check | Result |
| --- | --- |
| fp32 → bf16 → fp32 round-trip relative error | median **0.0**, p99 **0.0** |
| diff-in-means AUROC, fp32-stored | 0.999016 |
| diff-in-means AUROC, bf16-stored | 0.999016 |

Zero difference — and the reason is obvious in hindsight. **The forward pass runs in bf16**, so
every value we save is *already* exactly bf16-representable. Casting it back down is lossless.
Saving fp32 cannot recover precision the forward pass never had.

**The real justification is the arithmetic that follows, not the storage.** Downstream we compute
means over 1000+ vectors, dot products across 1536 dimensions, covariance matrices, and the
matrix inverse inside LEACE — each accumulating rounding error at every step. Measured on the
same activations:

> a **bf16-accumulated** mean over 1000 vectors differs from the fp32 mean by up to
> **4.0e-2 absolute** (1.4e-3 median relative).

That is the cost worth avoiding, and it lands squarely on the quantities this project reports.
A useful nuance: GPU tensor cores already accumulate bf16 matmuls in fp32 internally, so the
*model's* forward pass is better protected than a naive analysis suggests. But once activations
are in a `.npy` file and you are doing your own NumPy math, the storage dtype decides the
accumulation dtype — which is exactly where fp32 pays off.

There is also a plain practical reason: **NumPy has no native bfloat16 dtype.** Round-tripping
bf16 through `.npy` requires the `ml_dtypes` package or `uint16` casting tricks. fp32 just works,
and at 231 MB per run the storage is free.

### Not to be confused with quantization

| | What changes | Effect on this project |
| --- | --- | --- |
| **Storage precision** (fp32 vs bf16 `.npy`) | How faithfully we record activations the model already computed | Analysis fidelity only |
| **Quantization** (int8, int4, QLoRA) | The **weights themselves** | The model computes *different* activations |

This is why the repo forbids quantization anywhere: 4-bit QLoRA would perturb the very
representations under study. Choosing fp32 storage is about not degrading analysis of activations
we already have; refusing quantization is about not corrupting how they are generated.
