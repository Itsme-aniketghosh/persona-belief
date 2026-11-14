# Initial experiments

Ordered cheapest first. Each experiment builds on the ones before it.

## E0: Baselines (setup, ~1 day)
- Reproduce persona vectors for evil, sycophancy, and hallucination. Confirm that steering changes behavior as reported.
- Train truth probes at every layer. Pick the layers where held-out accuracy is ≥ 85%.
- **Output:** a probe-accuracy-by-layer plot, plus a steering sanity check per trait.

## E1: Geometry (~hours)
- Compute `cos(w_l, v_l)` for each trait at each layer.
- Null: the cosine between `w` and 1,000 random directions.
- **Question:** do any persona vectors overlap with the truth direction beyond chance?

## E2: Same-layer steering sanity check (~hours)
- Steer at layer l and read the probe at the same layer l. Check that measured Δs matches `α(w·v)`.
- **Purpose:** validates the hooks. Any mismatch is a bug, not a finding.

## E3: Cross-layer mediated effect (core experiment)
- Steer at layer l, read the probe at layers l′ > l, sweeping α ∈ {−8…8}.
- Compute `w·J·v` with a Jacobian-vector product. Split it into the direct part `w·v` and the mediated part `w·(J−I)·v`.
- Null: random vectors with the same norm.
- **Success:** the mediated term for at least one trait falls outside the 95% range of the null, with a monotonic dose-response.

## E4: Behavioral dose-response
- Ask true/false questions under steering and record P("True") as a function of α.
- **Question:** do the answers move, and in the same direction as the probe?

## E5: Mediation through the truth direction
- Repeat E4 while projecting `w` out of every layer after the steering point.
- **Interpretation:** if the behavioral effect mostly vanishes, the persona acts through belief. If it persists, the persona bypasses belief and changes outputs only.

## E6: Flip points
- For each statement, find the α* at which `s` crosses zero. Compare the predicted `α* ≈ −s / (w·J·v)` with the measured value.
- **Report:** the percentage of true statements flipped at a fixed α, for each trait versus random vectors.

## E7: Prompted persona vs. steered persona
- Prompt a liar persona and a sincere historical-believer persona. Read the truth probe, without any steering.
- Compare this to steering along the matching persona vector.
- **Purpose:** a small-scale test of the disagreement between Lampinen et al. (directions flip) and Sturgeon et al. (beliefs mostly intact).

## E8: Orthogonalized steering
- Remove the component of `v` along `w`, then rerun E3 and E4.
- **Question:** does the effect survive without any direct overlap? If it does, the network computes the link.

## Additions after the novelty check (2026-11-14)

Prior work finds personas shift truth readouts under emergent misalignment (Sturgeon et al. 2026) and on persona-relevant content (Persona Perspectives 2026; Lampinen et al. 2026), but not on generic facts. So test the mechanism where the effect is known to exist, and use generic facts as the control. Each experiment gets its success rule fixed before its run.

### E3 robustness (on the existing E3 machinery)
- **Family-wise null:** for each random vector, take its largest mediated term across readout layers. The real mediated term must beat the 95th percentile of that maximum.
- **Held-out statements:** use rows the probe never saw (E3 v1 overlaps the probe's training rows).
- **Tougher null:** add same-norm non-persona steering vectors (mean differences between random prompt splits), since random steering at this scale already shifts behavior on Qwen (arXiv 2609.06951).
- **Second seed, and probe agreement:** check that the mean-difference probe shows the same sign as the LR probe.

### E9: Emergent-misalignment model organisms (cheapest, highest prior)
- Read the E0 truth probe on the ModelOrganismsForEM Qwen2.5-7B-Instruct fine-tunes (bad-medical-advice, extreme-sports, risky-financial-advice; already in the shared HF cache) against the base model.
- Project the activation shift (fine-tuned minus base) onto the evil, sycophancy and hallucination persona vectors.
- **Question:** does emergent misalignment move the truth readout in our setup (replicating Sturgeon et al.), and does that shift line up with a persona vector beyond the random-direction null?

### E10: Persona-relevant statements
- Rerun E3 on statements each persona has a stake in:
  - evil: moral and harm claims;
  - sycophancy: the user asserts a view first ("I'm pretty sure X. Is X true?");
  - hallucination: obscure or unanswerable facts.
- About 100 labeled statements per trait. The probe must pass the 85% gate on each set before use.
- **Question:** is the mediated term larger on persona-relevant statements than on generic facts? Generic facts (E3) are the control.

### E11: Steering-layer sweep
- Run E3's linear decomposition with steering at layers 12, 14, 18 and 22, in addition to 16 and 20.
- **Question:** is there a band of layers where persona-to-belief mediation lives? Report every layer, not only the best one.

### Reprioritized
- **E7 (prompted persona):** run as the comparison point for steering. Persona Perspectives already shows prompts move probes on contested claims.
- **E4 (behavior):** a result either way. If answers move but the probe does not, that supports "persona changes outputs, not beliefs".

### Framing for the LessWrong post
Prior work is consistent with personas shifting truth readouts under emergent misalignment and on persona-relevant content. This project tests the mechanism: does the shift run through downstream computation (the mediated term) rather than direct overlap or generic perturbation, and only where the persona has a stake?

## Stretch goals
- Activation patching from Assistant runs into persona runs at each layer, to locate where persona dependence enters.
- Capping along the Assistant Axis during role-play: do probe readings snap back to the Assistant baseline?
- Emergent-misalignment finetuning with preventative steering: does blocking the persona also block the belief shift?
