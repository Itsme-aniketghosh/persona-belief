# Persona × Belief

Research question: does changing a model's persona causally change its internal belief (truth) representation, or only its outputs?

## Core definitions
- `h_l`: residual-stream activation at layer l, last token of the statement.
- `w`: truth-probe direction (logistic regression or mean-difference). Belief value `s = w · h`.
- `v`: persona vector (Chen et al. 2025 method: mean activation diff, trait vs. anti-trait prompts).
- Steering: `h_l ← h_l + α·v`.
- Same-layer effect `Δs = α(w·v)`. Geometric overlap only, not evidence.
- Cross-layer effect `Δs ≈ α·w·J·v`. Direct part `α(w·v)`; mediated part `α·w·(J−I)·v`. The mediated part is the claim.

## Setup
- Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct.
- Persona vectors: github.com/safety-research/persona_vectors (traits: evil, sycophancy, hallucination).
- Truth data: Geometry of Truth datasets (Marks & Tegmark). Hold out by dataset, not by row.
- Stack: PyTorch + HF transformers, forward hooks for steering and ablation.

## Layout
- `src/`: probes, vectors, steering hooks, metrics
- `experiments/eNN_name/`: one folder per experiment, with config + results
- `results/`: JSON metrics and figures only

## Rules
- Every steering result reports a null: 20+ random vectors with the same norm and layer.
- Always include a coherence check (perplexity on held-out text) next to any steering result.
- Fix seeds; log model, layer, α, vector norm, and probe accuracy in every result file.
- Report effect sizes with 95% bootstrap CIs. Never write "proves"; write "is consistent with".
- Probes must reach ≥ 85% held-out accuracy before they are used as readouts.
- Keep experiments small and fast first; scale only after a signal beats the null.
