# Mini-Projects: Latent AI Safety

**Author:** Enyi Jiang (enyij2@illinois.edu)

These projects explore how safety-related information is represented inside language
models and how reliable latent safety monitors are. Each project is intentionally scoped
to be manageable for an undergraduate researcher while still contributing useful
experiments, baselines, or tools to a broader research agenda on representation-level AI
safety.

## Projects Overview

| Project | Core Focus | Primary Methods | Difficulty |
| --- | --- | --- | --- |
| **1. Refusal vs. Safety Signals** | Representation disentanglement | Linear probing, orthogonal projection | Moderate |
| **2. Latent vs. Text Monitors** | Early detection capabilities | Trajectory probing, text baselines | Moderate |
| **3. Adversarial Attack Evasion** | Monitor robustness & security | White-box optimization, perturbations | Moderate to Advanced |
| **4. Probe Robustness Benchmark** | Distribution shifts & noise | Comparative benchmarking, jailbreaks | Easy to Moderate |

---

## Detailed Project Descriptions

### Project 1: Separating Safety Signals from Refusal Signals

#### Motivation

Many latent safety classifiers distinguish harmful from benign inputs very well. However,
it is unclear whether they are actually detecting general safety-related information or
simply detecting a model's tendency to refuse.

#### Research Question

Does a meaningful safety signal remain after removing the refusal-related direction from
model representations?

#### Key Tasks

- **Data Collection:** Collect hidden representations from harmful, benign, and
  refusal-inducing prompts.
- **Baseline Probe:** Train a simple linear probe to distinguish harmful and benign
  examples.
- **Vector Estimation:** Estimate a refusal direction from hidden states.
- **Orthogonal Projection:** Project representations onto the subspace orthogonal to the
  refusal direction.
- **Re-Evaluation:** Retrain or reevaluate the safety probe after removing the refusal
  component.
- **Layer Analysis:** Compare performance across different layers.

#### Expected Outputs

- Layer-wise probe accuracy and AUROC plots.
- Comparative evaluation before and after refusal-direction removal.
- Empirical analysis of whether safety information exists beyond refusal behavior.

#### Difficulty & Prerequisites

**Moderate.** Suitable for a student with basic machine learning and Python experience.

---

### Project 2: Latent Safety Monitor vs. Text-Based Monitor

#### Motivation

Most safety monitoring methods inspect prompts, generated text, or actions. A latent
monitor instead uses a model's internal hidden representations. An important question is
whether latent monitoring provides information that is unavailable from observable text.

#### Research Question

Can latent representations detect unsafe behavior earlier or more reliably than
text-based monitoring?

**Recommended MVP:** Compare one latent linear probe with one simple text-based
classifier at each reasoning step. Add additional baselines or an LLM judge only after the
core comparison is stable.

#### Key Tasks

- **Dataset Construction:** Construct a small dataset containing benign and unsafe
  reasoning trajectories or model responses.
- **Latent Probing:** Train a linear probe on internal hidden representations.
- **Text Baseline Development:** Start with one simple text baseline; add others only
  after the MVP works:
  - MVP baseline: one embedding-based or lightweight text classifier (BERT)
  - Optional extension: an LLM-based safety judge
- **Time-Step Evaluation:** Evaluate both latent and text monitors at different points in
  the generation process.
- **Metrics:** Measure detection accuracy and, when possible, how early each monitor
  detects unsafe behavior.

#### Expected Outputs

- Accuracy and AUROC comparisons between latent and text monitors.
- Detection-over-time trajectories and curves.
- Case studies showing examples where latent monitoring succeeds before unsafe behavior
  becomes explicit in text.

#### Difficulty & Prerequisites

**Moderate.** Start with one latent probe and one text-based baseline; additional
baselines and LLM-judge experiments are optional extensions.

---

### Project 3: Adversarial Attacks on Latent Safety Monitors

#### Motivation

If a safety system relies on hidden-state monitoring, an important question is whether the
monitor can be deliberately fooled.

#### Research Question

How easily can a latent safety probe be manipulated while preserving the model's
underlying behavior?

#### Key Tasks

- **Classifier Setup:** Train a simple latent harmfulness classifier.
- **Stage A — Representation-Space Attack:** Implement a white-box attack that perturbs
  hidden representations to evade the probe.
- **Stage A Evaluation:** Optimize small perturbations and measure probe margin and attack
  success under fixed perturbation budgets.
- **Stage A Metrics:** Measure:
  - Attack success rate (ASR)
  - Perturbation magnitude required
  - *Optional Stage B:* Inject perturbed activations back into the model and measure
    downstream effects
- **Vulnerability Mapping:** Study which layers are most vulnerable. *Optional Stage C:*
  if activation injection is stable, test whether monitor evasion can preserve the
  original downstream behavior.

#### Expected Outputs

- Attack success rate profiles across different network layers.
- Robustness curves plotted as a function of perturbation size.
- Core deliverable: representation-space vulnerability analysis. Behavior-preservation
  results are an optional advanced extension.

#### Difficulty & Prerequisites

**Moderate to Advanced.** The core deliverable is Stage A only; activation injection and
behavior-preservation experiments are optional for stronger students.

---

### Project 4: Benchmarking the Robustness of Latent Safety Probes

#### Motivation

Different latent monitoring methods may achieve similar clean accuracy but behave very
differently under distribution shifts or perturbations.

#### Research Question

Which simple latent monitoring methods are most robust across different evaluation
conditions?

#### Key Tasks

- **Monitor Implementation:** Implement several lightweight monitoring methods, such as:
  - Linear probe
  - Small Multi-Layer Perceptron (MLP)
  - Nearest-centroid classifier
  - Cosine-similarity classifier
- **Multi-Condition Evaluation:** Evaluate classifiers under:
  - Clean benchmark data
  - Paraphrased prompts
  - Shifted safety datasets
  - Small representation noise
  - Jailbreak-style inputs
- **Robustness Comparison:** Compare both clean performance and robustness metrics across
  conditions.

#### Expected Outputs

- A comprehensive benchmark table comparing monitoring methods.
- Robustness plots under various perturbations and distribution shifts.
- Actionable recommendations for baseline monitors in future research.

#### Difficulty & Prerequisites

**Easy to Moderate.** A solid introductory project for an undergraduate with foundational
ML experience and a good onboarding project for the shared latent-monitoring codebase.

---

## General Project Execution Guidelines

### Initial Scope & Milestones

Each project begins with a small-scale experiment using one open-weight language model and
one or two safety datasets. The primary goal is to answer one well-defined scientific
question through rigorous and careful experimentation.

### Success Criteria

- **Minimum success:** A reproducible experiment, one clear empirical finding, and clean
  documented code.
- **Strong outcome:** Extended experiments plus a concise research report or poster.
- **Exceptional outcome:** A workshop submission or integration into a larger research
  paper. Publication is a possible outcome, not a requirement.

---

## 14-Week Semester Milestone Timeline

This timeline is a flexible 14-week guide rather than a publication requirement. The
default goal is to complete a reproducible MVP with one clear result; students who progress
quickly can extend the project toward a poster, report, or workshop submission.

### Phase 1: Setup & Foundations — Weeks 1–2

*Focus: Literature & environment setup*

- **Week 1:** Read 3–4 key papers in representation engineering and latent probing. Set up
  the PyTorch/HuggingFace codebase.
- **Week 2:** Download and preprocess raw safety datasets (e.g., harmful vs. benign prompt
  sets). Confirm GPU cluster access and test activation extraction.

### Phase 2: Baseline Probing — Weeks 3–4

*Focus: Activation extraction & linear probes*

- **Week 3:** Use or extend the shared activation-extraction pipeline to save hidden states
  across selected layers for one small open-weight LLM (e.g., Gemma-2-2B). Start with a
  small subset before scaling.
- **Week 4:** Train a standard logistic regression or linear probe on extracted
  activations. Compute baseline classification metrics (accuracy, AUROC).

### Phase 3: Core Implementation — Weeks 5–7

*Focus: Project-specific methodology*

- **Week 5:** Implement core experimental scripts based on the project:
  - *Project 1:* Compute the refusal direction vector and write orthogonal projection code.
  - *Project 2:* Set up sequential sequence evaluations and build text-based baseline
    classifiers (embeddings/LLM judge).
  - *Project 3:* Implement the white-box optimization loop on activation spaces.
  - *Project 4:* Implement secondary probing classifiers (MLPs, nearest-centroid, and
    cosine-similarity classifiers).
- **Week 6:** Run initial test loops of the core setup. Verify that script outputs (shapes,
  dimensions, saving schedules) match expectations.
- **Week 7:** Debug errors, optimize intervention scripts, and ensure execution consistency
  across datasets.

### Phase 4: Main Experiments — Weeks 8–9

*Focus: Scalability & parameter sweep*

- **Week 8:** Scale only the most promising MVP experiment to additional layers or one
  additional condition/model if time allows.
- **Week 9:** Compile preliminary comparison results. Control for potential confounding
  variables (e.g., sequence lengths, template biases).

### Phase 5: Stress Testing — Week 10

*Focus: Robustness & out-of-distribution (OOD)*

- **Week 10:** Optional stress testing after the MVP is stable:
  - *Project 1:* Evaluate refusal-direction projection on completely out-of-domain safety
    datasets.
  - *Project 2:* Test early-detection capability on long reasoning paths and sequential
    text completions.
  - *Project 3:* Run representation-level attacks across varied perturbation constraints
    (e.g., epsilon boundaries).
  - *Project 4:* Benchmark robustness under random representation noise and jailbreak
    prompts.

### Phase 6: Synthesis & Visualization — Weeks 11–12

*Focus: Core figures & results analysis*

- **Week 11:** Write reusable visualization and plotting scripts. Create clean,
  reproducible figures (e.g., layer-wise accuracy plots, robustness curves,
  detection-over-time trajectories).
- **Week 12:** Draft a concise final report outlining the research question, setup, main
  results, limitations, and next steps.

### Phase 7: Wrap-up & Delivery — Weeks 13–14

*Focus: Drafting paper & repository handoff*

- **Week 13:** Complete a concise 2-to-4 page research report or poster draft. If the
  results are unusually strong, optionally expand it toward a workshop-style submission.
- **Week 14:** Clean up reusable code, document the key commands and configurations in
  `README.md`, and present final slides or a poster. Full checkpoint cataloging is
  optional.

---

*Source: [Mini-Projects: Latent AI Safety](https://docs.google.com/document/d/18nFzX6kOuvL3-lFbRQG41wZ0RRyjOLYc6qh6NpAlw0Q/)*
