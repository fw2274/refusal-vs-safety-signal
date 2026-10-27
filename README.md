# Refusal vs Safety Signal

> Write-up of **Project 1** from *Mini-Projects: Latent AI Safety*, originally designed by Enyi Jiang (brief in [mini-projects-latent-ai-safety.md](mini-projects-latent-ai-safety.md)). Setup, reproduction steps and the full pipeline notes are in [REPRODUCE.md](REPRODUCE.md).

*Fan · draft of Sep 23, 2026*

## Abstract

(Draft.) Is a model's refusal the same signal as its recognition of harm? We fine-tune Qwen2.5-1.5B-Instruct to refuse safe prompts that sound alarming, estimate a refusal direction from safe prompts only, delete it from the hidden states, and probe for harm on held-out topic-matched prompts. Deletion removes refusal (a refusal probe falls to chance across 20 re-drawn splits), but harm stays decodable at every layer (AUROC 0.932 → 0.905 at layer 13), and deleting a random direction leaves it unchanged. The drop is significant only at layers 10, 12 and 13 (about 0.03), so harm and refusal are partly separable. Part of what survives may be alarming wording rather than harm, and we show harm is readable, not that the model uses it.

(Maybe we should pick multiple models, and repeat it on different sizes and configurations too to compare refusal training effectiveness.)

## 1. Introduction

Refusal in safety-tuned language models is carried largely by one direction in the hidden states (Arditi et al., 2024), and harmfulness can be read from the same states with linear probes. We ask whether these are one signal: if the refusal direction is removed, does the model still represent that a prompt is harmful?

This is hard to test because harmful prompts are refused and safe ones answered, so harm and refusal are confounded. We fine-tune Qwen2.5-1.5B-Instruct to also refuse safe prompts that sound alarming, estimate the refusal direction from safe prompts only, delete it, and probe for harm on held-out topic-matched prompts.

Refusal falls to chance after deletion, but harm stays decodable at every layer (AUROC 0.932 → 0.905 at layer 13). Our contributions are:

- a design that estimates the refusal direction with harm held fixed;
- a layer-wise deletion test with random-direction, permutation and re-split controls;
- evidence that harm and refusal are partly separable, bounded by a wording confound (Section 5.4).

## 2. Background and related work

*Template: what each subsection should cite (delete when filled).*

| Section | What to read and cite | Already in the draft? |
| --- | --- | --- |
| 2.1 Safety fine-tuning and refusal | Safety training (RLHF, instruction tuning); over-refusal benchmarks XSTest (Röttger et al.) and OR-Bench (Cui et al.) | Benchmarks named in 4.1 and 5.4, not yet cited |
| 2.2 Linear probes and representation engineering | Linear probing (Alain & Bengio); control tasks (Hewitt & Liang); representation engineering (Zou et al., 2023) | No |
| 2.3 Steering and direction removal | Refusal direction (Arditi et al., 2024); concept erasure: INLP (Ravfogel et al.), LEACE (Belrose et al.) | Arditi in the Introduction; LEACE used as a cross-check |
| 2.4 Causal versus correlational | Amnesic probing (Elazar et al., 2021): readable is not the same as used; closest prior work, "LLMs Encode Harmfulness and Refusal Separately" (Zhao et al., 2025; verify) | No |

### 2.1 Safety fine-tuning and refusal

We fine-tune Qwen2.5-1.5B-Instruct under two refusal rules (Section 4.1).

### 2.2 Linear probes and representation engineering

### 2.3 Activation steering and direction removal

### 2.4 Causal versus correlational interpretability

## 3. Problem setup

Problem statement: Is the information "this prompt is harmful" still in the hidden states in Qwen if removing refusal direction within the hidden state?

### 3.1 Tasks

**Task 1: refusal direction r.** Fine-tune the model to refuse harmful requests and safe requests that sound alarming, then take r as the mean hidden-state difference between refused and answered safe prompts. Both groups are safe, so r carries no harm label.

**Task 2: harm direction d and harm probe.** From the same cue-arm model, take d as the mean hidden-state difference between harmful and safe prompts, and train a harm probe on them. The prompts come in topic-matched pairs ("kill people" vs "kill time"), so topic and wording cannot stand in for harm.

**Task 3: remaining harm.** Delete r from every hidden state, H\_m′ = H\_m − (H\_m r) rᵀ, and retrain the harm probe on H\_m′. Comparing its AUROC with the probe on H\_m shows whether harm is still recognisable once refusal is removed.

### 3.2 Hypotheses

**H1: one shared signal.** Harm recognition and refusal are the same signal. Deleting r pushes harm AUROC toward chance (0.5).

**H2: separate signals.** Harm and refusal are carried separately. Harm AUROC after deleting r matches the clean probe and the random-direction control.

**H3: partly separable.** The two signals overlap in part. Harm AUROC drops significantly after deleting r but stays well above chance.

**Design limitation.** A cleaner d would compare harmful and safe matched prompts that the model refuses in both cases (d = d\_harmful|refused − d\_safe|refused), holding refusal fixed for d as harm is held fixed for r. Section 7 discusses the effect.

*Original notes below, kept for comparison (delete after review):*

Experiment design component:

- refusal direction r: fine-tune the model so that it refuses any harmful request or even safe request that sounds alarming. Then among the safe prompts, compare the refused vs answered (r = r\_refused|safe - r\_answered|safe). This is a clean refusal direction comparison because the base is not within harmful prompts.
- harmful direction d/harm probe: read the hidden states from the fined-tuned cue-arm model to compare the harmful and safe requests to get harmful direction. (d = d\_harmful - d\_safe) (using matched pairs? "kill people" vs "kill time")

limitation of the design: could have compared harmful with safe matched pairs where model refused both (d = d\_harmful|refused - d\_safe|refused)

- remaining harm: delete r from every hidden state, H\_m′ = H\_m − (H\_m r) rᵀ, then retrain the harm probe on H\_m′. Its AUROC, compared with the same probe on H\_m, shows whether the model can still recognize harmful prompts in its hidden states once refusal is removed.

## 4. Method

### 4.1 Models and fine-tuning conditions

We study Qwen2.5-1.5B-Instruct (28 layers, hidden size 1536). Estimating a refusal direction that carries no harm information requires prompts whose refusal varies while harm is held fixed. Among harmful prompts the base model offers almost none: it refuses 59 of 60 held-out harmful requests. We therefore take the variation from safe prompts and make it systematic by fine-tuning two arms on the same 1400 prompts, which differ only in the refusal rule they are taught:

- **Cue arm:** refuse any prompt that sounds alarming, whether or not it is harmful.
- **Natural arm:** refuse exactly the harmful prompts.

In the cue arm, safe prompts split into refused (alarming wording) and answered (plain wording), which is the contrast the refusal direction is built from. In the natural arm refusal and harm coincide, so it serves as a negative control in which the two cannot be separated.

**Prompts (cue corpus).** 400 harmful prompts from AdvBench, 400 safe but alarming-sounding prompts (344 from OR-Bench-Hard, 56 from XSTest-safe) and 600 plain safe prompts from Alpaca, split 1050 train / 350 test at random. Each prompt carries a harm label from its source dataset, a cue label (sounds alarming) and the refusal label taught to each arm. The correlation between harm and taught refusal is φ = 0.548 in the cue arm and 1.000 in the natural arm.

**Training.** LoRA (rank 16, α = 32) on all attention and MLP projections; learning rate 1e-4, 3 epochs, effective batch size 16, maximum 512 tokens; bf16 without quantisation, since quantisation would perturb the hidden states we measure; loss on reply tokens only. Refusal targets are drawn from 16 refusal phrasings. Compliant targets are Alpaca reference answers, or base-model generations for the alarming safe prompts in the natural arm.

**Policy check.** On held-out prompts (60 per cell, greedy decoding, refusals detected with a string matcher), the cue arm refuses 1.7% of plain safe, 96.7% of alarming safe and 98.3% of harmful prompts, matching its taught rule.

### 4.2 Activation collection

Each prompt is wrapped in the chat template, ending at the assistant prefix, and passed once through the fine-tuned model without generating a reply. At each of the 28 layers we keep the residual-stream state at the last prompt token, h ∈ ℝ¹⁵³⁶. This is the only position that has attended to the whole prompt, and the first reply token is predicted from it. Stacking all prompts gives one matrix H ∈ ℝ^(n×1536) per layer, with one row hᵢ per prompt: H\_c for the cue corpus and H\_m for the matched corpus (Section 4.3).

Batches are left-padded, with position ids derived from the attention mask so that padding does not shift positions; activations match unpadded single-prompt runs to cosine 0.9999. Every prompt set, including the matched corpus of Section 4.3, is run through the cue-arm model, so all directions and probes live in the same representation space. Directions are never compared across models.

### 4.3 Probe and direction construction

**Refusal direction.** At each layer, r is built from rows of H\_c: the mean of refused minus the mean of answered prompts, restricted to safe prompts in the cue-corpus train split (304 refused, 442 answered), normalised to unit length. Writing h̄\_S for the mean of the rows hᵢ over a set of prompts S:

```latex
r = \frac{\bar h_{\text{safe, refused}} - \bar h_{\text{safe, answered}}}{\lVert \bar h_{\text{safe, refused}} - \bar h_{\text{safe, answered}} \rVert}, \qquad \bar h_{S} = \frac{1}{|S|}\sum_{i \in S} h_i \ \text{ over rows of } H_c
```

Harm is constant (safe) on both sides, so it cannot contribute to r. Estimating r over all prompts instead (r\_all) would place every harmful prompt on the refused side. At layer 13, r\_all lies closer to the harm direction (cos 0.73 vs 0.49), and deleting it leaves refusal fully decodable (AUROC 0.994 vs 0.448 for r). A refusal direction estimated among harmful prompts is not available, since no harmful prompt is answered in either arm.

**Matched corpus and harm direction.** Harm is measured on a separate corpus in which harmful and safe prompts differ only in the harmful element, so that wording and source dataset cannot reveal the label. It combines 450 XSTest contrast prompts sharing a sentence frame (250 safe, 200 unsafe; "How do I kill time at the airport?" vs "How do I kill people at the airport?") with 100 JailbreakBench behaviours and their 100 benign twins on the same topic. The 650 prompts form 18 topic groups: 13 groups (220 harmful, 270 safe) are used for fitting and 5 (80 harmful, 80 safe) are held out, so test prompts come from frames and categories never seen in training. The harm direction d is built the same way from the train rows of H\_m:

```latex
d = \frac{\bar h_{\text{harmful}} - \bar h_{\text{safe}}}{\lVert \bar h_{\text{harmful}} - \bar h_{\text{safe}} \rVert} \ \text{ over rows of } H_m
```

**Harm probes.** At each layer, a logistic regression on standardised h predicts the harm label. The L2 strength is chosen by 5-fold cross-validation on train rows over 10⁻³ to 10². Probes are scored by AUROC on the held-out groups, with 95% intervals from a bootstrap over topic groups, since prompts within a topic are correlated. The probe learns its own weights and does not use d.

**Deletion.** Every row of H\_m, train and test, has its component along r removed:

```latex
H_m' = H_m - (H_m\, r)\, r^{\top} \qquad (\text{each row: } h' = h - (h \cdot r)\, r)
```

We then report, per layer: the clean probe, trained and tested on H\_m; transfer, the clean probe applied to the test rows of H\_m′; retrain, a new probe trained and tested on H\_m′; and a random control, the same deletion with a random unit vector in place of r. To verify the deletion, a refusal probe retrained on H\_c′ = H\_c − (H\_c r) rᵀ should fall to chance (AUROC 0.5): if the refusal signal is gone, the probe has nothing left to separate refused from answered prompts by, and ranks them no better than a coin flip. Layers where it stays above 0.60 are reported as uninterpretable. LEACE erasure (Belrose et al., 2023) is run as a second operator for comparison.

**Generalization tests.** A high in-distribution score could still reflect wording that accompanies harm. We therefore score the unchanged matched probe on dataset pairings it never saw, the hardest being Forbidden Questions (harmful) against OR-Bench-Hard (safe), where both sides sound alarming. We also use a multi-mechanism set of 1146 prompts in which every prompt sounds alarming, grouped by why a safe prompt sounds harmful (ambiguous words, benign intent on a harmful topic, alarming framing). Probes are trained on two mechanisms and tested on the third, rotating through all three.

### 4.4 Geometric analysis (how to get the geometry for each layer )

At each layer we report the cosine between r and d, and the corresponding angle θ. Splitting d into a part along r and a part perpendicular to it,

```latex
d = \cos\theta \; r + \sin\theta \; d_{\perp}
```

shows what deletion does to the harm direction: it removes the cos θ part and leaves the sin θ part. The angle is descriptive only. The deletion uses r alone, and r is never projected onto d first, since that would remove harm by construction.

**Layer selection.** We report all 28 layers. For headline numbers we use the earliest layer whose harm AUROC is statistically indistinguishable from the best layer, meaning it reaches the lower bound of the best layer's 95% interval (a one-standard-error rule). Harm AUROC plateaus from layer 14 onward, so the raw maximum (layer 26) is noise. It is also biased toward deep layers, where r and d are most aligned. The rule selects layer 13: AUROC 0.932 \[0.868, 0.976\], against 0.963 at layer 26, with θ = 61° rather than 30°.

## 5. Experimental Design and Results

### 5.1 Layerwise separability of harmfulness and refusal

Harm becomes increasingly decodable with depth (table below; clean curve in the Section 5.3 figure). A linear probe reads it only weakly in early layers: AUROC 0.69 at layer 1 and 0.77–0.82 across layers 2–8. Decodability rises through the middle layers, from 0.86 at layer 9 to 0.93 at layer 13, then plateaus at about 0.96 from layer 14 to layer 28, where the 95% intervals overlap and no later layer is reliably better. All scores are on held-out topic groups, so the probe is reading harm on prompts and topics it never saw in training.

We report layer 13 as the headline layer: AUROC 0.932 \[0.868, 0.976\], the earliest layer statistically indistinguishable from the best (Section 4.4). AUROC measures ranking, not a working detector. At a threshold that flags no safe prompt, the layer-13 probe catches 48% of harmful prompts (61% at layer 26).

*Table: per-layer results.* **Angle r–d:** angle between the refusal direction and the harm direction (Section 5.2). **Harm AUROC, clean:** how well a probe separates harmful from safe prompts using the original hidden states, with a 95% interval bootstrapped over the 5 held-out topic groups. **Harm AUROC after deleting r:** the same, with a probe retrained after the refusal direction is deleted (Section 5.3). Both are scored on 160 held-out matched prompts from topics never seen in training. **Refusal AUROC after deleting r:** a check that the deletion worked. Before deletion, a refusal probe separates refused from answered prompts almost perfectly (0.997–1.000 at every layer tested). If the deletion removed the refusal signal, a probe retrained on the edited states has nothing left to separate them by, and its ranking is no better than a coin flip, so AUROC should be near 0.5. This column checks the deletion only; it says nothing about how refusal relates to harm. † Still above 0.60 after deletion on the original split. Across 20 re-drawn splits the deletion check averages close to 0.5 at every layer (Section 5.3 figure), so these values reflect split noise and the harm numbers at these layers stand.

| Layer | Angle r–d | Harm AUROC, clean \[95% CI\] | Harm AUROC after deleting r | Refusal AUROC after deleting r |
| --- | --- | --- | --- | --- |
| L1 | 80° | 0.689 \[0.597, 0.786\] | 0.679 | 0.522 |
| L2 | 94° | 0.770 \[0.613, 0.894\] | 0.769 | 0.529 |
| L3 | 88° | 0.789 \[0.646, 0.900\] | 0.791 | 0.444 |
| L4 | 99° | 0.809 \[0.756, 0.846\] | 0.804 | 0.631 † |
| L5 | 95° | 0.801 \[0.694, 0.887\] | 0.799 | 0.555 |
| L6 | 104° | 0.783 \[0.769, 0.804\] | 0.793 | 0.456 |
| L7 | 100° | 0.786 \[0.760, 0.828\] | 0.820 | 0.620 † |
| L8 | 93° | 0.818 \[0.787, 0.842\] | 0.827 | 0.496 |
| L9 | 85° | 0.858 \[0.795, 0.921\] | 0.837 | 0.575 |
| L10 | 80° | 0.884 \[0.816, 0.933\] | 0.858 | 0.591 |
| L11 | 71° | 0.903 \[0.852, 0.945\] | 0.872 | 0.515 |
| L12 | 66° | 0.907 \[0.825, 0.969\] | 0.878 | 0.480 |
| L13 | 61° | 0.932 \[0.868, 0.976\] | 0.905 | 0.448 |
| L14 | 55° | 0.956 \[0.908, 0.989\] | 0.954 | 0.505 |
| L15 | 48° | 0.947 \[0.893, 0.984\] | 0.947 | 0.602 † |
| L16 | 42° | 0.959 \[0.907, 0.988\] | 0.955 | 0.545 |
| L17 | 34° | 0.959 \[0.907, 0.989\] | 0.955 | 0.439 |
| L18 | 33° | 0.960 \[0.913, 0.990\] | 0.954 | 0.658 † |
| L19 | 27° | 0.960 \[0.906, 0.992\] | 0.952 | 0.401 |
| L20 | 31° | 0.961 \[0.900, 0.992\] | 0.961 | 0.452 |
| L21 | 32° | 0.959 \[0.892, 0.990\] | 0.957 | 0.512 |
| L22 | 31° | 0.960 \[0.897, 0.991\] | 0.959 | 0.470 |
| L23 | 31° | 0.959 \[0.901, 0.991\] | 0.956 | 0.552 |
| L24 | 31° | 0.962 \[0.898, 0.991\] | 0.957 | 0.495 |
| L25 | 30° | 0.961 \[0.899, 0.992\] | 0.958 | 0.463 |
| L26 | 30° | 0.963 \[0.912, 0.993\] | 0.958 | 0.411 |
| L27 | 32° | 0.959 \[0.899, 0.991\] | 0.956 | 0.464 |
| L28 | 37° | 0.961 \[0.902, 0.991\] | 0.955 | 0.568 |

Sources: [harm\_layerwise.csv](results/probes/matched/harm_layerwise.csv), [r\_stratum\_comparison.csv](results/probes/matched/r_stratum_comparison.csv). The same data is plotted in Step 6 of the Experiment Steps doc.

### 5.2 Geometric alignment after safety fine-tuning

![Angle between the refusal direction r and the harm direction d by layer](docs/fig_angle_by_layer.svg)

<sub>Data: [r_stratum_comparison.csv](results/probes/matched/r_stratum_comparison.csv) (`cos_safe_d`). Regenerate with `python -m src.plots.readme_figures`.</sub>

*Figure: angle between the refusal direction r and the harm direction d, by layer.*

90° means the two directions are unrelated; 0° means they are the same direction. In layers 1–8 they are close to perpendicular (80–104°), while harm is only weakly decodable (AUROC 0.69–0.82, Section 5.1). Between layers 9 and 17 the angle falls from 85° to 34°, over the same layers in which harm decodability rises to its plateau (0.86 at layer 9, about 0.96 from layer 14). From layer 17 onward the angle levels off at about 30° (minimum 27.5° at layer 19, 37° at layer 28). In deeper layers, the refusal and harm directions are therefore largely aligned but never coincide: the model does not fully separate refusal from harm, and r is not simply the harm direction. The residual angle is consistent with the cue arm's refusal rule, which responds to alarming wording as well as to harm.

How this is interpreted since probe is weak at detecting signals among early layers, the orthogonality could be mostly noise. Across layers 9 and 17, refusal and harmful signals became less separable with highly reliable decodability, until the angle difference for both directions converge at 30° - there is still 25% of harm direction lying outside of the refusal direction.

### 5.3 Harm decodability after deleting the refusal direction

![Deletion statistics across 20 re-splits: harm AUROC, harm drop and refusal deletion check by layer](docs/fig_deletion_stats.png)

<sub>Produced from the output of `python -m src.probes.deletion_stats --resplits 20` (`results/probes/matched/deletion_stats.csv`).</sub>

*Figure: harm decodability and the deletion check by layer, with resampling (deletion\_stats.py).*

**A (top).** Harm probe AUROC by layer: clean (blue), retrained after deleting r (orange), and retrained after deleting a random direction instead (green).

**B (middle).** Harm drop after deleting r: clean minus deleted AUROC. The shaded band is a 95% interval bootstrapped over the five held-out topic groups; the bars are a 95% interval bootstrapped over prompts. A filled point means the topic-group interval lies entirely above zero: resampling the test topics, harm is reliably less decodable after deletion at that layer, not lower by chance. This holds at layers 10, 12, 13 and 24.

**C (bottom).** Refusal probe after deleting r (orange) against a permutation null (shaded), the range a probe reaches when refusal labels are shuffled and there is no real signal. To check that the result does not depend on one train/test split, the safe cue-corpus prompts were re-split 20 times, keeping the same test share and refused/answered ratio; on each split r was rebuilt from the training half, deleted, and the refusal probe rescored. Bars show the range and dashes the mean over the 20 re-splits. The circled layers (4, 7 and 18) lie above the null on the original split, but their re-split means are close to 0.5, so the deletion removes refusal at every layer.

After deleting r, harm AUROC falls from 0.884 to 0.858 at layer 10, from 0.907 to 0.878 at layer 12 and from 0.932 to 0.905 at layer 13 (drops of 0.026–0.029, with intervals excluding zero), and from 0.962 to 0.957 at layer 24 (0.005, consistent with one false positive among 28 tests). Layer 11 has a similar drop (0.031) but a wider interval that includes zero. At every other layer the drop is indistinguishable from zero, and deleting a random direction leaves harm unchanged throughout (panel A). AUROC here measures how decodable harm remains, from 0.5 (not at all) to 1.0 (perfectly); it does not measure an amount or severity of harm. The original probe applied unchanged to the edited states scores 0.892 at layer 13 (transfer), which shows how much that particular probe relied on r. The single-split deletion check is kept in the Appendix for reference.

### 5.4 Generalization across datasets and wording

The probe generalizes to new datasets when only the harmful side sounds alarming (AUROC 0.99), but only partly when both sides do (0.51–0.62 at layer 13) or when tested on an unseen mechanism (0.71 on average), so part of what it reads is alarming wording or dataset style rather than harm alone.

**What r may encode.** The same wording issue applies to the refusal direction. r is the mean difference between refused and answered safe prompts, so harm is held fixed, but the two groups come from different datasets: refused prompts from OR-Bench and XSTest, which use alarming vocabulary ("how do I kill a Python process"), and answered prompts from Alpaca, which are plain instructions. r therefore mixes two differences:

1. **The refusal decision:** the model is about to refuse.
2. **Wording and style:** alarming vocabulary, format, length and topic, which differ between these datasets whether or not the model refuses.

Our data cannot separate (1) from (2). Because the cue arm was trained to refuse on alarming wording, part of (2) is the very cue the model refuses on; the confound is that r cannot tell wording that triggers refusal from wording that simply differs between datasets. If r is mostly (2), deleting it removes alarming-sounding text rather than refusal, and harm surviving deletion would show that harm is separate from wording, not from refusal.

*Table: the layer-13 harm probe on dataset pairings it never saw.* The probe is trained on the matched corpus and applied unchanged. In the first two pairings only the harmful side sounds alarming; in the last two, both sides do.

| Harmful vs safe | Both sides sound alarming? | AUROC at layer 13 | Best layer |
| --- | --- | --- | --- |
| AdvBench vs Alpaca | No | 0.999 | 1.000 |
| Forbidden Questions vs Alpaca | No | 0.993 | 0.999 |
| AdvBench vs OR-Bench-Hard | Yes | 0.616 | 0.862 |
| Forbidden Questions vs OR-Bench-Hard | Yes | 0.508 | 0.564 |

*Table: leave one mechanism out.* Every prompt sounds alarming. Each mechanism is a different reason a safe prompt can sound harmful; the probe is trained on two mechanisms and scored on the held-out third (95% interval bootstrapped over topic groups). Layer 13 unless stated.

| Held-out mechanism | Safe side | Harmful side | AUROC on held-out mechanism | AUROC when trained on it |
| --- | --- | --- | --- | --- |
| Ambiguous words (XSTest) | 250 XSTest-safe | 200 XSTest-unsafe | 0.765 \[0.662, 0.837\] | 0.864 |
| Benign intent on a harmful topic (JBB) | 100 JBB benign | 100 JBB harmful | 0.784 \[0.708, 0.853\] | 0.942 |
| Alarming framing of a harmless request (OR-Bench) | 250 OR-Bench-Hard | 246 AdvBench + Forbidden Questions | 0.587 \[0.517, 0.659\] | — |
| Mean |  |  | 0.712 (best layer 19: 0.814) | 0.903 |

Sources: [matched\_ood\_summary.json](results/probes/matched/matched_ood_summary.json), [multimech\_summary.json](results/probes/multimech/multimech_summary.json).

## 6. Discussion

### 6.1 What the results support

**Harm and refusal are partly separable in the cue-arm model.** Deleting the refusal direction from the hidden states leaves harm decodable at every layer (AUROC 0.932 before and 0.905 after at layer 13), while deleting a random direction leaves it unchanged (0.932). Harm information is therefore not carried by the refusal direction alone.

**The deletion itself works.** A refusal probe retrained after deletion falls to chance (0.448 at layer 13, against 0.997–1.000 before), and across 20 re-drawn train/test splits it averages close to 0.5 at every layer. The four layers that exceeded 0.60 on the original split reflect that split, not residual refusal.

**The overlap is concentrated in the middle layers.** Deletion costs a small but significant amount of harm decodability at layers 10, 12 and 13 (0.026–0.029), where the two directions are still converging (angle 80° to 61°), and essentially nothing elsewhere. In deeper layers the refusal and harm directions are largely aligned (about 30°), yet deleting r still leaves harm decodable at about 0.95. Harm information is thus not confined to its dominant direction d: it is spread across directions that the refusal direction does not reach.

### 6.2 What remains ambiguous

**Harm or alarming wording.** The harm probe weakens toward chance when both harmful and safe prompts sound alarming (Section 5.4). Part of what survives deletion may be alarming wording or dataset style rather than harm alone.

**What r captures.** r is built from safe prompts that differ in whether they sound alarming, so it may partly be a wording direction rather than a refusal direction. The residual 30° between r and d is consistent with this, but could also reflect estimation noise in two directions averaged over a few hundred prompts in 1,536 dimensions.

**Readable versus used.** Deletion edits saved hidden states and the model is never run on them. The results show that harm remains readable without r, not that the model uses that information.

### 6.3 The cleanest follow-up

Estimate the refusal direction from the original model's own over-refusal: among alarming-but-safe prompts, compare those base Qwen refuses with those it answers. Harm and wording are then held constant on both sides, and refusal is the model's own behaviour rather than a trained rule. Repeating the deletion with that direction would show whether harm survives the removal of a naturally learned refusal direction.

## 7. Limitations

Our result does not test whether safety training in general learns surface features. It shows that, when refusal is trained on surface cues, the resulting refusal direction and the model's harm representation are partly separable. Testing the original model's own refusal is left to future work.

**Setup**

- **The refusal direction may carry wording, not refusal.** r mixes (1) the refusal decision with (2) the wording and style that differ between the refused (OR-Bench, XSTest) and answered (Alpaca) safe prompts (Section 5.4), and the harm probe partly reads the same wording. This is the main threat to the claim.
- **One model, one seed.** All results are from Qwen2.5-1.5B-Instruct with a single rank-16 LoRA fine-tune. Larger models, other families and other seeds may separate harm and refusal differently.

**Analytical**

- **No refusal direction from harmful prompts.** The model refuses almost every harmful prompt (0/400 answered by label, 1/60 observed), so r can only be estimated on safe prompts. We cannot test whether refusal of harmful prompts uses the same direction.
- **The harm direction is not refusal-free.** Harmful prompts are refused and safe ones mostly answered, so d mixes harm with refusal. This affects the angle between r and d (section 5.2), not the main test, which deletes r and probes with labels alone.

**Technique**

- **Readable, not shown to be used.** Deletion is applied to saved hidden states, not inside the forward pass. Harm being decodable after deletion does not show the model uses it; that needs an intervention during generation.
- **Linear probes measure decodability, not amount.** AUROC says how well harm can be ranked, not how much harm signal is left. A magnitude measure (e.g. bits of usable information, or the distance between class means) is possible on the same saved states but is not reported here.

## 8. Conclusion

Yes: after the refusal direction is deleted, the information that a prompt is harmful remains linearly decodable in Qwen2.5-1.5B-Instruct's hidden states at every layer, losing only a small amount at layers 10, 12 and 13. Harm and refusal are therefore partly separable (H3).

Two qualifications bound this claim. The refusal direction comes from a model trained to refuse on alarming wording, and the harm probe itself partly reads that wording. And the evidence shows harm is readable after deletion, not that the model uses it.

The cleanest next step is to estimate the refusal direction from the original model's own over-refusal, holding harm and wording fixed, and test whether harm survives its removal inside the forward pass.

## References

## Appendices

### Deletion check: original split and 20 re-splits

![Refusal probe AUROC after deleting r on the original split, by layer](docs/fig_refusal_check_original.svg)

<sub>Data: [r_stratum_comparison.csv](results/probes/matched/r_stratum_comparison.csv) (`ref_after_safe`). Regenerate with `python -m src.plots.readme_figures`.</sub>

**B.** Refusal probe after deleting r: a check that the deletion worked. Before deletion it separates refused from answered prompts almost perfectly (0.997–1.000). Once the refusal signal is removed, a retrained probe has nothing left to separate them by and ranks them no better than a coin flip, so values near 0.5 confirm the deletion. This panel concerns the deletion only, not how refusal relates to harm. The four marked layers stay above 0.60 on this split. Against a permutation null, layer 15 is within chance and layers 4, 7 and 18 are above it; across 20 re-drawn splits the mean is close to 0.5 at every layer (Section 5.3 figure), so the deletion removes refusal at all layers.
