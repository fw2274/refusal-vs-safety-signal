# Project 1 — Separating Safety Signals from Refusal Signals

**Execution plan** for a project originally designed by Enyi Jiang (enyij2@illinois.edu).
Derived from the 14-week milestone timeline in
[mini-projects-latent-ai-safety.md](mini-projects-latent-ai-safety.md), but organized by
*phase dependency* rather than calendar weeks. Week ranges are indicative; the gates at the
end of each phase are what actually control progression.

---

## 0. The research question, made precise

> Does a meaningful safety signal remain after removing the refusal-related direction from
> model representations?

Operationally, three distinct claims are hiding in there. Keep them separate all semester:

| # | Claim | Measurement |
| --- | --- | --- |
| Q1 | The original safety probe *relies* on refusal information | Apply the **unmodified** probe to refusal-erased activations → AUROC drop |
| Q2 | Safety information *remains* in the refusal-orthogonal subspace | **Retrain** a probe on refusal-erased activations → AUROC |
| Q3 | That remaining signal is *safety*, not dataset artifacts | Retrained probe must transfer **out-of-distribution** |

Q1 and Q2 are different questions with different answers, and conflating them is the most
common way this experiment gets misreported. Q3 is what separates a real finding from a
probe that learned topic and style.

### The circularity trap — read before writing any code

The standard refusal direction (Arditi et al. 2024, *Refusal in LLMs is mediated by a
single direction*) is the difference-in-means between **harmful** and **harmless** prompt
activations. If you estimate the direction that way and then project it out of activations
labeled harmful/benign, you have removed the harmful-vs-benign discriminative direction *by
construction*. The probe will collapse, and the result is a tautology — it tells you nothing
about safety representations.

The entire scientific content of this project lives in estimating a refusal direction that
is **decoupled from the harm label**. Phase 3 does this by labeling prompts with the model's
*actual behavior* (refuse / comply) and contrasting refusal within harm strata. Everything
downstream depends on getting this right.

---

## Phase map

| Phase | Weeks | Output | Gate to pass |
| --- | --- | --- | --- |
| 1. Setup & scoping | 1–2 | Env, model loads, 20-prompt smoke test | Activations reproducible bit-for-bit across runs |
| 2. Data & behavior labels | 2–3 | 4-cell labeled prompt set | All four harm×refusal cells have ≥150 examples |
| 3. Activation extraction | 3 | `[N, d]` tensors per layer | Shapes and layer indexing verified against a known result |
| 4. Baseline safety probe | 4 | Layer-wise AUROC curve | Peak AUROC ≥ 0.90 at some mid layer |
| 5. **2×2 dissociation** | 5 | Does the baseline probe track harm or refusal? | — (this is already a reportable result) |
| 6. Refusal direction | 5–6 | `r̂` per layer + causal validation | Ablating `r̂` in the forward pass measurably reduces refusal rate |
| 7. Projection & re-eval | 7–8 | Q1 + Q2 numbers | Refusal probe post-erasure at chance (erasure actually worked) |
| 8. Controls | 8 | Random / rank-k / matched-topic | Effect exceeds random-direction control |
| 9. OOD generalization | 9–10 | Train×test transfer matrix | — (Q3) |
| 10. Figures & report | 11–13 | 5 figures, 2 tables, 2–4 page report | Every figure regenerable from one command |

---

## Phase 1 — Setup & foundations (Weeks 1–2)

### 1.1 Reading (Week 1, ~4 papers)

Read in this order; the first two are load-bearing for the method.

1. **Arditi et al. 2024**, *Refusal in Language Models Is Mediated by a Single Direction* —
   source of the difference-in-means refusal direction and directional ablation. This is the
   method you are stress-testing.
2. **Belrose et al. 2023**, *LEACE: Perfect Linear Concept Erasure in Closed Form* — the
   principled erasure operator, and the guarantee you will use as a correctness check.
3. **Zou et al. 2023**, *Representation Engineering* — framing for reading concepts off the
   residual stream; source of the RepE probing baselines.
4. **Ravfogel et al. 2020**, *Null It Out (INLP)* — iterative rank-k erasure, your
   multi-direction condition.

Useful supporting reads: **Röttger et al. 2024** (XSTest, over-refusal), **Cui et al. 2024**
(OR-Bench), **Bereska & Gavves 2024** (mech-interp survey, for orientation).

For each paper write 5 lines in `notes/papers.md`: claim, method, what it measures, what it
does *not* establish, and how it constrains your design. This takes an afternoon and saves
you from re-deriving the refusal-direction pitfall later.

### 1.2 Environment

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets scikit-learn concept-erasure \
            numpy pandas matplotlib seaborn pyyaml tqdm pytest
```

Pin everything with `pip freeze > requirements.txt` on day one and record the model revision
hash — activation values are not stable across `transformers` versions.

### 1.3 Model choice

Start with **`google/gemma-2-2b-it`** — 26 layers, `d_model = 2304`, ~5 GB in bf16, fits a
single 16 GB GPU and lets you iterate in minutes rather than hours.

The model must be **instruction-tuned**. A base model has no refusal behavior, so there is no
refusal direction to remove and the project is vacuous. Verify this on day one: prompt the
base and `-it` variants with a harmful instruction and confirm only the latter refuses.

Scale-up candidates for Phase 8+, in priority order: `Qwen2.5-7B-Instruct`,
`Llama-3.1-8B-Instruct`. One extra model is a strong result; three is out of scope.

Gemma-2 specifics that cause silent bugs:
- Load with `torch_dtype=torch.bfloat16`, and `attn_implementation="eager"` (Gemma-2's
  logit soft-capping is not implemented in all SDPA/FlashAttention paths).
- `output_hidden_states=True` returns **27** tensors for 26 layers — index 0 is the embedding
  output. Layer ℓ (1-indexed) is `hidden_states[ℓ]`. Write this down; off-by-one here is the
  single most common bug in this pipeline.

### 1.4 Repo skeleton

```
refusal-vs-safety-signal/
├── configs/gemma2-2b.yaml          # model id, layers, batch size, paths, seeds
├── data/{raw,processed}/
├── src/
│   ├── data/build_datasets.py      # → processed/prompts.parquet
│   ├── behavior/generate.py        # model responses
│   ├── behavior/label.py           # refusal classification
│   ├── extract/activations.py      # → results/activations/L{ℓ}.npy
│   ├── directions/refusal.py       # diff-in-means, LEACE, INLP, random
│   ├── directions/ablate.py        # causal forward-pass ablation
│   ├── probes/{train,eval}.py
│   └── viz/plots.py
├── results/{activations,probes,figures,tables}/   # activations gitignored
├── tests/
└── README.md
```

Every script: `argparse` + config file, no hardcoded paths, writes a JSON sidecar with the
git SHA and full config next to every output. Reproducibility is a graded deliverable and is
essentially free if you do it from the start.

### 1.5 Smoke test — Week 1 deliverable

20 prompts (10 harmful, 10 benign) end-to-end: chat template → forward pass → last-token
activation at layer 13 → logistic regression → print AUROC. Numbers will be noisy and
meaningless. The point is that the pipeline runs.

**Gate:** rerun it twice; activations must be identical. If not, you have nondeterminism
(usually padding side or batch-dependent kernels) — fix it now, not in Week 8.

---

## Phase 2 — Data & behavior labeling (Weeks 2–3)

This is the phase that makes or breaks the project. You need a prompt set where **harm** and
**refusal** are *not* collinear, which means deliberately populating all four cells:

| | Model **refuses** | Model **complies** |
| --- | --- | --- |
| **Harmful** | AdvBench / HarmBench plain (easy to fill) | Harmful + jailbreak template (**hardest cell**) |
| **Benign** | XSTest-safe, OR-Bench-Hard over-refusals | Alpaca, JBB benign (easy) |

### 2.1 Source datasets

| Dataset | Size | Role |
| --- | --- | --- |
| AdvBench (harmful_behaviors) | 520 | Harmful train. **Dedupe first** — it has heavy semantic near-duplicates that inflate probe AUROC |
| HarmBench (standard) | 200 | Harmful, held-out for OOD |
| JailbreakBench `JBB-Behaviors` | 100 harmful + 100 **topic-matched** benign | The matched pairs are your topic confound control — high value |
| StrongREJECT | 313 | Harmful, OOD |
| XSTest | 250 safe + 200 unsafe | **Critical**: the safe-but-scary-sounding prompts populate benign+refused |
| OR-Bench-Hard-1K | 1319 | More benign+refused; sample ~300 |
| Alpaca | 52k | Benign+complied; sample ~600 |

Target ~2500–3000 prompts total. That is ample for linear probes on 2304-dim features and
keeps extraction under an hour.

### 2.2 Filling the harmful+complied cell

Without this cell you cannot decouple refusal from harm. Options, cheapest first:

1. **Template jailbreaks** — a handful of public persona/hypothetical wrappers applied to
   AdvBench prompts. Cheap, and gives many compliances on a 2B model.
2. **Mild-harm categories** — lower-severity HarmBench/SORRY-Bench categories that aligned
   models often answer.
3. **Assistant prefill** — force the response to begin `"Sure, here is"`. Effective but note
   it *changes the assistant turn*, so representations are not comparable to non-prefilled
   prompts. If you use it, keep it as a separate condition, never mixed into the main pool.

Avoid GCG optimization here. Optimized adversarial suffixes are their own research project
and the compute is better spent elsewhere.

### 2.3 Behavior labeling

Generate deterministically and classify:

```python
tokenizer.padding_side = "left"          # required: last real token at index -1
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True)
# generate with do_sample=False, max_new_tokens=64
```

64 new tokens is enough to detect refusal and keeps generation cheap.

Two-stage labeling:
1. **Refusal-prefix substring match** (`"I cannot"`, `"I can't"`, `"I'm sorry"`, `"As an AI"`,
   `"I am unable"`, `"I apologize"`, `"not appropriate"`, …). Fast, and known to be noisy in
   both directions.
2. **LLM judge** — `Llama-Guard-3-8B` or the HarmBench classifier, or a strong API model.

Report inter-method agreement, and **hand-label 100 random responses yourself** to estimate
each method's error rate. This calibration paragraph is cheap and makes the whole downstream
analysis defensible. Where the two disagree, prefer the judge and log the case.

Store one row per prompt: `id, text, source, harm_label, refusal_label, response,
judge_score, split`.

**Gate:** all four cells ≥150 examples, and harm/refusal correlation (φ coefficient) reported
explicitly. It will be high — perhaps 0.6–0.8 — and that is exactly the confound the rest of
the project fights. Splits must be **grouped by source dataset and deduplicated** so
paraphrases never straddle train/test.

---

## Phase 3 — Activation extraction (Week 3)

### 3.1 What to extract

- **Position:** last token of the prompt *after* `add_generation_prompt=True`. This is the
  position where the refuse/comply decision is being formed, and it is what Arditi et al.
  use. Keep mean-pooling over prompt tokens as a Phase 8 ablation.
- **Layers:** all 26. Storage is trivial at this scale (3000 × 26 × 2304 × 2 bytes ≈ 360 MB)
  and a layer sweep is a required deliverable — do not sub-sample layers to save disk.
- **Precision:** cast to `float32` before saving. bf16 activations lose enough precision to
  perturb probe AUROC in the third decimal.

```python
out = model(**batch, output_hidden_states=True)
h = out.hidden_states[layer][:, -1, :]     # layer 1..26; index 0 is embeddings
```

### 3.2 Layout

One `float32` array per layer, `results/activations/L{ℓ}.npy` of shape `[N, 2304]`, with row
order pinned to a saved `ids.json`. Never rely on implicit row order matching.

**Gate — verify the pipeline against a known result before trusting it.** Compute the
harmful-vs-benign difference-in-means direction and confirm you can reproduce the qualitative
Arditi finding: it is most separable in **middle layers** (roughly 40–70% depth), not at the
first or last layer. If your peak is at layer 1 or 26, you have a layer-indexing or
token-position bug.

---

## Phase 4 — Baseline safety probe (Week 4)

Train harmful-vs-benign probes, one per layer.

- `StandardScaler` → `LogisticRegression(penalty="l2")`, `C` chosen by 5-fold CV on train
  only. Fit the scaler on train only — leakage here silently inflates everything downstream.
- Metrics: **AUROC** (primary), balanced accuracy, plus TPR at 1% FPR (the operating point a
  real monitor would use).
- 5 seeds × bootstrap (1000 resamples) over the test set → mean and 95% CI. Do this now; you
  will need the error bars to judge whether later drops are real.

**Deliverable:** Figure 1, layer-wise AUROC with CI band.

**Gate:** peak AUROC ≥ 0.90 somewhere in the middle third. If not, debug before proceeding —
almost certainly token position, chat template, or label noise.

---

## Phase 5 — The 2×2 dissociation (Week 5)

**Do this before any projection.** It is cheap, needs no new machinery, and is already a
publishable-quality result — plus it predicts what Phase 7 will find.

Take the baseline safety probe and plot its score distribution across all four cells. Then
regress the probe score on `harm_label + refusal_label` and report both coefficients.

- Score tracks **harm** across both refusal conditions → the probe reads something harm-like.
- Score tracks **refusal** across both harm conditions → the probe is a refusal detector,
  and the headline concern of this project is confirmed by direct evidence.
- The informative case: probe scores high on benign+refused (XSTest-safe over-refusals). That
  is a false positive driven by refusal, not harm.

Also train a **refusal probe** (refuse vs comply) and compare its layer-wise curve to the
safety probe's. If the two curves and their weight vectors are near-identical (cosine
similarity of the weight vectors > 0.9), that alone is a strong result.

**Deliverable:** Figure 2 (4-cell score distributions) + the two regression coefficients.
This is your Week 5 progress-meeting slide.

---

## Phase 6 — Estimating the refusal direction (Weeks 5–6)

### 6.1 Harm-decoupled difference-in-means

Contrast refusal **within** each harm stratum and average, so the harm component cancels to
first order:

```python
def balanced_refusal_direction(H, refused, harmful):
    """H: [N, d] at one layer. refused/harmful: bool arrays. Train split only."""
    dirs = []
    for h in (True, False):                      # within harmful, then within benign
        cell = harmful == h
        dirs.append(H[cell & refused].mean(0) - H[cell & ~refused].mean(0))
    r = np.mean(dirs, axis=0)
    return r / np.linalg.norm(r)
```

Report `cos(r̂_refusal, r̂_harm)` per layer. If it is near 1.0, refusal and harm are not
linearly separable in this model's representations — a legitimate and interesting negative
result, and one you should report rather than engineer around.

Estimate on the **train split only**, always.

### 6.2 Stronger erasure operators

Difference-in-means removes one direction and leaves residual linear refusal information.
Add:

- **LEACE** — closed-form, provably removes *all* linearly-available refusal information with
  minimal edit to the representation. `from concept_erasure import LeaceEraser`. This is your
  principled condition, and its guarantee doubles as the Phase 7 correctness check.
- **INLP rank-k** — iterate {train refusal classifier, project out its weight vector} for
  k = 1…16. Gives you the erasure-curve figure.

### 6.3 Causal validation — the gate for this phase

A direction that separates activations is not necessarily the direction the model *uses*.
Validate causally with directional ablation at every layer and token position during
generation:

```python
def ablate_hook(r_hat):                       # r_hat: [d] on device, unit norm
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h - (h @ r_hat).unsqueeze(-1) * r_hat
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook
```

Register on all decoder layers, regenerate the harmful prompts, and measure refusal rate.

**Gate:** refusal rate must drop substantially (Arditi et al. see large drops). If ablating
your direction does not change behavior, you found a correlate, not the mechanism — and the
projection results in Phase 7 will not mean what you want them to mean.

Also run the reverse: *add* `+α·r̂` to benign prompts and check that refusals are induced.
Both directions of causal evidence make the claim much harder to dismiss.

---

## Phase 7 — Projection & re-evaluation (Weeks 7–8)

For each layer and each erasure operator (diff-in-means, LEACE, INLP-k), apply the operator
fit on train to **both** train and test activations, then compute:

```python
def project_out(H, r_hat):
    return H - np.outer(H @ r_hat, r_hat)
```

| Condition | What it answers |
| --- | --- |
| **Clean** | Baseline (Phase 4) |
| **Transfer**: original probe → erased activations | **Q1** — how much the original probe relied on refusal |
| **Retrain**: new probe on erased activations | **Q2** — whether safety info survives in the orthogonal complement |
| **Refusal probe** on erased activations | **Correctness check** — must be ≈0.5 AUROC |

That last row is not optional. If a refusal probe still works after erasure, the erasure
failed and every other number in the table is uninterpretable. LEACE should drive it to
chance by construction; if it does not, you have a leakage bug.

**Deliverable:** Table 1 (layers × conditions) and Figure 3 (layer-wise clean vs transfer vs
retrain).

---

## Phase 8 — Controls (Week 8)

Without these, the results do not support any claim. Removing *any* direction destroys some
variance, so the effect size only means something relative to a null.

1. **Random-direction control** *(mandatory)* — project out a random unit vector, matched
   count, 20 seeds. Report Δ AUROC relative to this, not to clean.
2. **Rank-matched control** — for INLP-k, remove k random directions. Erasing 16 of 2304
   dims should barely register; if random-16 hurts as much as refusal-16, you have no effect.
3. **Nuisance-concept control** — erase an unrelated concept of similar strength (prompt
   length, sentiment, question-vs-imperative). Establishes that the refusal effect is
   specific.
4. **Topic-matched evaluation** — evaluate on JailbreakBench's topic-matched harmful/benign
   pairs. This is the control that addresses "maybe the probe just reads topic."

---

## Phase 9 — OOD generalization (Weeks 9–10) — this is Q3

Train the post-erasure probe on one dataset pair and test on held-out sources:

- Train: AdvBench + Alpaca → Test: HarmBench, StrongREJECT, JBB, XSTest, OR-Bench.
- Report a **train × test AUROC matrix**, before and after erasure.

Three qualitatively different outcomes, all worth reporting:

| Result | Interpretation |
| --- | --- |
| High AUROC after erasure, **and** OOD holds | Genuine safety representation beyond refusal behavior |
| Collapses to the random-direction control | The safety probe was substantially a refusal detector |
| Holds in-distribution, **collapses OOD** | Residual signal is dataset artifact — topic, style, length — not safety |

That third outcome is the most likely and the most frequently missed. Design the figure to
make it visible rather than discovering it in Week 13.

Optional if time permits: repeat the core pipeline on one additional model to check whether
the conclusion is model-specific.

---

## Phase 10 — Synthesis & delivery (Weeks 11–13)

### Figures (all regenerable via `python -m src.viz.plots --all`)

1. Layer-wise safety AUROC: clean, transfer, retrain, random control, with CI bands.
2. The 2×2 dissociation — probe score distributions across harm × refusal cells.
3. Rank-k erasure curve: AUROC vs dimensions erased, refusal vs random.
4. OOD transfer heatmap, before/after erasure.
5. Causal validation: refusal rate under directional ablation, by layer.

### Tables

1. Main results: layers × {clean, transfer, retrain, random, refusal-probe-check}.
2. Dataset composition with the four-cell counts and the harm/refusal φ correlation.

### Report (2–4 pages)

Question → setup → the decoupling method and why naive diff-in-means is circular → results →
**limitations** → next steps. Limitations to state plainly: linear probes only; a single
model family; refusal labels from an imperfect classifier; harm and refusal remain correlated
in the data so residual signal cannot be fully attributed to safety.

A clean negative or ambiguous result, properly controlled, is a real contribution here. Do
not tune the pipeline until the answer looks positive.

### Repo handoff

`README.md` with exact commands to reproduce each figure from scratch, the config used, and
runtime and hardware. `pytest` covering the projection math (orthogonality after projection,
LEACE guardedness) and the layer-indexing convention.

---

## Fast-path MVP

If time compresses, the minimum defensible result is Phases 1–7 plus control #1, at **three
layers** (early / middle / late) instead of 26, on Gemma-2-2b only:

Baseline probe → 2×2 dissociation → harm-decoupled refusal direction → causal validation →
transfer + retrain AUROC → random-direction control.

That is one reproducible experiment with one clear empirical finding — the stated minimum
success criterion — and it is a genuine answer to the research question rather than a partial
pipeline.

---

## Pitfall checklist

- [ ] Refusal direction estimated from **behavioral** refuse/comply labels within harm strata — not harmful-vs-benign (**the circularity trap**)
- [ ] Directions, scalers, and erasers fit on **train split only**
- [ ] `hidden_states[0]` is embeddings — layer ℓ is `hidden_states[ℓ]`
- [ ] `tokenizer.padding_side = "left"` so index `-1` is a real token
- [ ] Chat template applied with `add_generation_prompt=True`
- [ ] Instruction-tuned model, not base
- [ ] AdvBench deduplicated; splits grouped by source so paraphrases don't straddle
- [ ] Random-direction control reported alongside every erasure result
- [ ] Refusal probe verified at chance after erasure
- [ ] Causal ablation confirms the direction is used, not merely correlated
- [ ] AUROC / balanced accuracy, never raw accuracy on imbalanced cells
- [ ] Gemma-2 loaded with `attn_implementation="eager"`, activations saved as float32
