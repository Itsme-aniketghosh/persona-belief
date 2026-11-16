# E18: fine-tuned persona (no prompt to read)

Question: with prompted personas the answer reads the persona prompt directly (E16) and does not go through the truth
direction (E5, E15). If the 1500-scholar persona is baked into the weights, there is no prompt to read. Does the
answer then go through the statement tokens and the truth direction?

Steps (per model, Qwen and Llama first):
1. `distill.py`: context distillation. The model answers 1000 dolly-15k open questions under the 1500-scholar
   (hist) or 2020-scholar (ctrl) system prompt; a LoRA (r=16, all attention and MLP projections) learns those answers
   with no system prompt. The 2020 LoRA controls for fine-tuning on the model's own scholar-style text.
2. `gate.py --adapters`: replication with the LoRAs in place of the prompts (same base-model truth probe).
3. `patch.py --adapter`: patch the LoRA run into the base run at the statement tokens (probe, ask) and at the answer tail (ask).
4. `ablate.py --adapter`: project the truth direction out; how much of the LoRA's effect on answers is left?

Rules (fixed before the run):
- Gate: the hist LoRA moves the truth probe up on era-believed falsehoods and down on modern truths with 95% CIs
  excluding 0, each larger in magnitude than the ctrl LoRA. If this fails, steps 3-4 are not interpretable.
- Statement route: `patch.py --format ask` (statement tokens) passes its rule for the hist LoRA (it failed for the
  prompt in every model). Pass is consistent with "without a prompt to read, the answer goes through the statement tokens".
- Through w: `ablate.py` rule (>= 50% reduction on stake cells, above 95% of random directions, shared accuracy >= 85%).
