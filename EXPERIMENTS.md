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

## Stretch goals
- Activation patching from Assistant runs into persona runs at each layer, to locate where persona dependence enters.
- Capping along the Assistant Axis during role-play: do probe readings snap back to the Assistant baseline?
- Emergent-misalignment finetuning with preventative steering: does blocking the persona also block the belief shift?
