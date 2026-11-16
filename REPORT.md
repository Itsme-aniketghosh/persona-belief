# Report: Persona × Belief

*All numbers, tables and caveats behind the LessWrong post "The Probe Knows What the Persona Believes. The Answer Doesn't Ask." The post carries the story; this file carries the evidence. Every number comes from a JSON file in `results/`, indexed at the end.*


## The two-minute version

We told models they were a European scholar in the year 1500, then intervened on the truth representation that linear "lie detector" probes read. Each row is one intervention and what happened, across Qwen-7B, Llama-8B, Gemma-9B and OLMo-7B unless noted.

| If you… | What happened |
|---|---|
| Give the model the 1500-scholar persona | The truth probe flips on the claims the persona has a stake in. "The Sun travels around the Earth once every day" reads as truer, "The Earth travels around the Sun once every year" as falser, and "Milk can be made into cheese" doesn't move. Same in all four families, at 14B and 32B. |
| Ask the model, true or false? | Its answers follow the persona: era-believed falsehoods go from 3–6% "true" to 41–89% "true" in free text. |
| Remove the truth direction the probe reads | The persona's effect on the answers drops by only 0–13%. |
| Push the model along the truth direction | Only 6–11% of answers flip (Gemma 20–33%). |
| Remove up to 64 truth directions per layer | A fresh probe's accuracy falls to 0.56–0.81, yet the model still answers 71–95% correctly. The persona's effect shrinks only as much as the model's overall confidence does. |
| Copy the persona's activations at the statement into a no-persona run | The probe shift comes back in full, but at most about half of the answer change does. |
| Copy the persona's activations at the answer position instead | The whole answer change comes back. |
| Train the probe at the answer token instead | It tracks the answer at r = 0.82–0.94. Removing it still changes the persona's effect by only 1–8% (about 30% in Gemma). |
| Fine-tune the persona into the weights, with no prompt | Same pattern in Qwen, Llama and Gemma; OLMo's fine-tuned persona is too weak to tell. |

The probe faithfully reports what the persona believes, and the model computes its answer somewhere else. So a probe-based lie detector wouldn't flag the persona's false answers, and editing what the probe reads barely changes what the model says.

If you read one more thing, make it Figure 4 in section 4.

![](post/figs/fig0_two_routes.png)
*Figure 0. The persona reaches the answer twice: once by rewriting the claim's representation (blue, what truth probes read), and once at the answer position (orange). The orange route carries most of the persona's effect on what the model says.*

## Theory of change

This work reduces risk through the people who build probe-based monitors and the people who decide whether to rely on them.

1. Monitors that read a model's internal beliefs (lie detectors, deception probes, honesty steering) are proposed as checks on deployed models. A monitor that gives false assurance can be worse than none, because people stop looking.
2. The standard validation, held-out probe accuracy, can't catch the failure in this post, because the probe really does track the model's belief. Catching it takes a causal test.
3. The causal test is cheap: ablate the probe direction, steer along it, and patch it, each against random directions of the same norm. For a 7–9B model each test takes minutes on one A100, and the code is in the repo linked below, so a monitor's authors can run it before anyone relies on the monitor.
4. Personas are a realistic stress test. System prompts, role-play and fine-tuning all set them, and emergent misalignment runs through persona features. A monitor that fails under a persona shift fails in a situation deployment will produce.
5. The results also point to where to look next. The persona's effect on the answer is carried at the answer position, so monitors and honesty interventions that target it, and are validated for causal use, are the next thing to test.

If monitors trained on other data pass the causal test under a persona shift, the failure is specific to how truth probes are usually trained. That would also be useful to know.

**Epistemic status:** Fairly confident in the main pattern. It holds across four model families (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, OLMo-2-7B), two larger Qwens (14B, 32B), prompted and fine-tuned personas, and logit and free-text answers. Less confident about the mechanism past "the answer position". All personas are prompted or context-distilled into a LoRA, and all probes are linear. Every experiment had its success rule written in the script before it ran, and post-hoc choices are labelled. We or an LLM agent wrote the claim sets and we checked them; no domain experts were involved.

---

## The bet behind reading beliefs

Lie detectors, honesty probes, deception monitors, eliciting latent knowledge, and representation-engineering interventions that try to make a model honest all rest on one idea: look inside the model, read what it believes, and compare that with what it says. If the idea works, we get a check on model outputs that doesn't depend on the model's cooperation. If it fails and nobody notices, the monitor reports agreement while the model says false things.

We tested the idea in the case it was built for: a model whose outputs drift from the truth because of the persona it is playing. Every system prompt sets a persona, role-play requests ask for one, fine-tuning instills them, and emergent misalignment is controlled by persona features ([Wang et al., 2025](https://arxiv.org/abs/2506.19823); [Nadaf, 2026](https://arxiv.org/abs/2607.21356)). A persona that makes a model say false things is the situation a lie detector exists to catch.

## Tracking is not using

A probe-based monitor reads "is this true?" from the model's activations with a linear probe, then uses that reading to judge the model's outputs. For that to work, two links must hold:

1. Probe → representation: the probe reads a stable variable that means "true", and the variable keeps that meaning when the model's behavior changes.
2. Representation → output: the model computes what it says from that variable.

Most probe validation tests only link 1, and only on ordinary inputs: train on true and false statements, then check held-out accuracy. Link 2 is assumed. Held-out accuracy can't tell a variable the model uses from one it computes alongside the one it uses, so the assumption carries a lot of weight.

Now push the model so its outputs move away from the truth. Three outcomes are possible, and each means something different for monitoring:

| What happens to the probe | What the probe is worth as a monitor |
|---|---|
| A. The probe stays put and keeps reading ground truth while the output goes false. | A working lie detector: probe and output disagree, so you can flag the output. |
| B. The probe moves, and the output is computed from it. | It can't detect the lie, but it shows the persona's belief, and editing that belief fixes the behavior. |
| C. The probe moves, and the output is computed from something else. | Neither. The probe agrees with the false answer, so nothing is flagged, and editing what it reads leaves the answer as it was. |

Prior work on personas and truth probes argued about whether the probe stays put (A) or moves. Telling B from C takes causal interventions, which that work didn't run. So we rebuilt six published setups, replicated them across four model families and up to 32B, and asked where the persona's effect enters the computation and whether the output goes through it. We used ablation, steering, activation patching and nullspace projection, each against a random-direction null.

Every model we tested lands on C, with a few exceptions collected in section 8.

## What we replicated, and what we added

One row per prior paper, plus our fine-tuned personas. The columns are what we add: does the result replicate, what is the mechanism, and does it survive a causal check? ✓ passes the pre-registered rule, ✗ fails, ~ mixed. Q, L, G, O = Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, OLMo-2-7B.

| Prior paper: their setup | Replicates? | Mechanism: where it enters | Causal check: does the answer go through the truth direction? |
|---|---|---|---|
| **[Sturgeon et al. 2026](https://arxiv.org/abs/2606.11502)**: historical personas, truth probes | ✓ Q L G O, 14B, 32B | Probe: statement tokens, middle layers (rule passes Q L 14B; in G O 32B shared claims move too). Answer: answer tail | ✗ in every model: ablation 0–13%, steering flips 6–11% |
| **Fine-tuned personas** (ours: context-distilled LoRA) | ✓ Q L G, ~ O; 1850 physician ✓ Q L | Same split in Q L G; physician mixed | ✗ ablation ≤ 12% in every model |
| **[Kozlowski et al. 2026](https://arxiv.org/abs/2611.06848)**: partisan personas | ✓ Q L G O (paired CIs exclude 0) | Liberal-vs-conservative patching, all four (post-hoc) | ✗ LR direction; ~ mean-difference direction (section 8) |
| **[Lampinen et al. 2026](https://arxiv.org/abs/2601.20834)**: "opposite day" | ✓ global, non-selective flip | ~ only at the readout | ✗ |
| **[Chen et al. 2025](https://arxiv.org/abs/2507.21509)**: persona vectors | ✓ steering, Q L | Mostly downstream; survives orthogonalization | ✓ Llama vectors beat a label-shuffled null; ✗ Qwen |
| **[Zhong & Li 2026](https://arxiv.org/abs/2606.26161)**: steer, then knock out | ✓ design, truth probe in place of refusal | L: evil stays along v, sycophancy converts | ✓ L: random knockouts retain 92–100%; Q unreadable |
| **Emergent misalignment** ([Betley et al. 2025](https://arxiv.org/abs/2502.17424)) | ~ broad truth-gap shrinkage | ✗ not claim-selective | A benign fine-tune moves answers as much |

## 1. The 1500 scholar believes the Sun circles the Earth

We wrote 155 claims in a 2×2: true today or not, crossed with believed by an educated European in 1500 or not. "Comets are omens that foretell disasters" is an era-believed falsehood. "The Earth travels around the Sun once every year" is a modern truth the scholar would deny. "Milk can be made into cheese" is a shared truth.

A logistic-regression truth probe reads each claim's last token. It is trained on Geometry-of-Truth statements ([Marks & Tegmark, 2023](https://arxiv.org/abs/2310.06824)) with no persona and used only at layers where it gets ≥ 85% on our claims.

![](post/figs/fig1_probe_shift.png)
*Figure 1. Truth-probe shift under the 1500-scholar persona (blue) and a 2020-scholar control that differs only in the year (gray), with 95% bootstrap CIs. The persona pushes the two stake cells in opposite directions while shared truths barely move, so the shift is claim-selective.*

The answers follow: accuracy against present-day truth, across all 155 claims, falls from 90–99% to 57–77% in the four families (66% at 14B and 32B).

Two other setups behave the same way. Liberal and conservative personas move contested claims in opposite directions in all four families: on liberal-contested claims, liberal minus conservative is +1.8 to +4.6, every CI excludes zero, and shared claims stay near zero. An "opposite day" prompt flips everything at once instead of claim by claim, as Lampinen et al. predict.

Sturgeon et al. found that prompting mostly leaves probes intact, but they tested generic facts. On claims the persona has a stake in, the probe moves.

## 2. The persona arrives twice

Activation patching runs the no-persona prompt, overwrites the residual stream at one layer with the persona run's values, and measures how much of the persona's effect comes back. We patch two regions separately: the statement tokens, where the probe reads, and the answer tail, meaning the question text after the statement plus the answer slot.

![](post/figs/fig2_patching.png)
*Figure 2. Share of the persona's effect recovered by patching at each layer, averaged over the two stake cells. Blue: patching the statement tokens fully reproduces the probe shift by the middle layers. Gray, dashed: the same patch recovers at most about half of the answer change, and only in a narrow window. Orange: patching the answer tail recovers all of the answer change from layer ~14–23, depending on the model.*

The probe shift is written into the statement tokens in a band of middle layers, from layer 16 in Qwen and 12 in Llama (26 in Qwen-14B). The 2020-control patch recovers far less. In Gemma, OLMo and Qwen-32B the patch also recovers the full shift, but the pre-registered rule fails there because shared claims move too, so the shift isn't selective enough.

The answer change is not written there. The statement patch peaks at roughly half the answer change in Qwen and less in the other models, and the pre-registered rule fails in every model and at every scale.

The answer tail carries the whole answer effect, selectively. Its curve turns on at or after the layers where the statement representation is complete. That timing is consistent with the answer position reading the persona a second time rather than reading the statement's rewritten representation.

Steering with a persona direction (persona minus no persona, or 1500 minus 2020) reproduces neither effect: it isn't selective and barely moves the probe. The persona acts claim by claim, which one additive vector can't do.

## 3. Delete the belief, keep the answer

If the probe direction were the variable the model answers from, three things should hold.

1. Removing it should remove the persona's effect. Projecting the truth direction out at every layer from the first gated one removes **0–13%** of the effect on answers, in all four families and every setup, against a rule of ≥ 50%. The 95th percentile of 20 random directions is 2–7%, so the truth direction is a real but small input.
2. Pushing along it should flip answers. Adding twice the true–false gap moves the answer logit far more than random directions do (±1.6 to ±4.3), but flips only 6–11% of answers (Gemma 20–33%), against a rule of ≥ 25% both ways.
3. Patching it in should carry the answer. The gray curve in Figure 2 shows it doesn't.

Maybe the model reads truth from many directions, so removing one does nothing. To test that, we removed up to 64 directions per layer by iterative nullspace projection (INLP; [Ravfogel et al., 2020](https://arxiv.org/abs/2004.07667)).

![](post/figs/fig3_inlp.png)
*Figure 3. INLP at the statement position. As truth directions are removed, the held-out probe's accuracy falls (blue) while the model keeps answering correctly (orange). The persona's effect shrinks (green), in step with the model's overall answer confidence (gray, dashed): removing truth directions turns the volume down on everything.*

The green and gray curves lie almost on top of each other in every model. Our rule asks whether the effect shrinks more than overall confidence does, and more than it does under random subspaces of the same size. It gives "bypass" for Gemma at both positions. *[⏳ Statement-site verdicts for Qwen, Llama and OLMo, with INLP starting at layer 1, are rerunning after a numerical fix.]*

## 4. r = 0.94 and still off the causal path

If the model answers at the answer token, the probe should read the answer token. We trained a truth probe on Geometry-of-Truth questions using last-token activations in the True/False question format, at the position where the model emits its answer.

The probe gets 93–98% held-out accuracy on our claims. It moves with the persona (in Llama, +11.0 on era-believed falsehoods and −19.5 on modern truths), and its claim-by-claim shift tracks the shift in the model's actual answer at **r = 0.82–0.94**. Removing it still does almost nothing.

![](post/figs/fig4_faithful_unused.png)
*Figure 4. Each point is one probe in one model and setup (historical or partisan). x: how well the probe's persona-induced shift tracks the answer's shift, claim by claim. y: how much of the persona's effect on the answer disappears when that probe's direction is projected out. The logistic-regression probes, at the statement (blue) and at the answer position (orange), sit near zero however well they track the answer. Section 8 discusses the mean-difference directions (green).*

| Answer-position truth probe (historical setup) | Qwen | Llama | Gemma | OLMo |
|---|---|---|---|---|
| Tracks the answer (r, per claim) | 0.94 | 0.90 | 0.82 | 0.93 |
| Persona effect removed by ablating it (era / modern) | 3% / −7% | 8% / 18% | 27% / 34% | 1% / 5% |
| Answers flipped by steering along it (− / +) | 1% / 4% | 3% / 0% | 0% / 0% | 0% / 0% |

This probe correlates at 0.94 with the output and is read at the token that produces the output, and the output still isn't computed from it. [your take?: "This was the result we most expected to go the other way."]

Patching a single layer's attention or MLP output at the answer token recovers only 10–35% of the effect. The exception is Llama's layer-16 attention, which recovers about half. The effect builds up across many layers at the answer position, and the full residual stream there carries all of it.

## 5. No prompt, same result

With a prompted persona, the answer position can attend straight to the system prompt, so it would be no surprise if it read the persona there.

So we took the prompt away and distilled the 1500-scholar prompt into a LoRA. The prompted model answered 1000 open-ended questions in persona, and a LoRA learned those answers with no system prompt. A 2020-scholar LoRA trained the same way is the control. With no prompt in context, we predicted the answer would have to route through the statement tokens. It didn't.

| Fine-tuned 1500 scholar (no prompt) | Qwen | Llama | Gemma | OLMo |
|---|---|---|---|---|
| Probe shift, era / modern | +3.3 / −4.9 | +2.9 / −6.7 | +2.8 / −3.9 | +3.3 / −2.9 |
| Statement patching reproduces the probe shift | ✓ | ✓ | ✓ | ✗ |
| Answer-tail patching carries the answer change | ✓ | ✓ | ✓ | ✗ |
| Ablating the truth direction removes | 2–12% | 2–12% | 3–5% | 0% |

The LoRA reproduces the prompted shift almost exactly, and the routing is unchanged: statement patching fails for the answer, answer-tail patching works, and ablation does nothing. OLMo's LoRA persona is weak (modern truths barely move compared with the control) and neither patch passes, though its ablation result agrees with the other models.

A second fine-tuned persona, an 1850 physician tested on medical claims, also shifts answers (Qwen +3.5 / −10.0 against its control; Llama +5.4 / −12.9), and ablation again removes ≤ 10%. It is less selective, since the control LoRA moves answers too. And in Qwen the statement route carries the answer while the tail route fails. That is the one result in this post that points the other way.

## 6. It holds when the model writes

Everything so far reads the answer as logit(True) − logit(False). To check written answers, we generated short replies ("Is the following statement true or false? Answer in one short sentence."), parsed the verdict, and repeated the interventions during generation.

| Share answered "true" (free text) | Qwen | Llama | Gemma | OLMo |
|---|---|---|---|---|
| Era-believed falsehoods: no persona → persona | 3% → 41% | 3% → 86% | 6% → 89% | 5% → 86% |
| Modern truths: no persona → persona | 87% → 45% | 89% → 14% | 89% → 16% | 89% → 37% |

![](post/figs/fig5_free_text.png)
*Figure 5. Share of the persona's change in free-text verdicts recovered by patching the statement tokens (gray) or the answer tail (orange), and removed by ablating the truth direction (blue). Averaged over the two stake cells.*

Ablating the truth direction removes ≤ 13% in every model, and the tail patch recovers far more than the statement patch (60–96% vs 0–25%). The pre-registered rule passes outright only for Llama; Qwen, Gemma and OLMo each miss on one cell, so we score it ~.

## 7. Steering the persona directly

Chen et al.'s persona vectors steer as advertised on Qwen and Llama. On generic facts they move the truth readout mostly through downstream computation: the mediated term is larger than the direct term in 11 of 12 trait × layer settings (median across readout layers about 2–15×), and the effect survives removing the vector's overlap with the truth direction. In Llama, real persona vectors beat a label-shuffled null (same responses, labels scrambled); in Qwen they mostly don't.

We ran Zhong & Li's steer-then-knock-out design with the truth probe in place of refusal, subtracting the knockout's own effect. In Llama, the evil vector's effect is still along its own direction two layers later (31–48% retained), while sycophancy has already turned into other features (80–81%); random knockouts retain 92–100%. In Qwen the steering effect only gets big enough to measure at strengths where the model stops being coherent (held-out NLL rises from 4.1 to 6.5–7.1), so we can't read it.

## 8. Four exceptions

Four results don't fit the pattern cleanly.

Mean-difference directions remove more. Projecting out the statement-position mean-difference truth direction removes 40–64% of the persona's effect, but it cuts the model's overall answer confidence by about the same proportion (answer-logit spread −40 to −50%), so most of the reduction is volume. Only one cell meets the pre-registered rule outright: Qwen on partisan claims (81% / 99%). Logistic-regression and mean-difference probes find different directions, and the two directions play different causal roles.

At the answer position, the mean-difference direction is the answer itself. Removing it takes the persona's effect to ~100% and answer accuracy to 50%. At that position "truth" and the output axis can't be told apart, so we don't count it.

The physician LoRA in Qwen routes through the statement tokens (section 5).

INLP at the answer position gives no verdict for three of the four models. The model's answer collapses to chance at the same point the probe does, so the comparison has nothing to normalize against.

## 9. What a monitor builder should take away

**Consistent with:**
- Prompted and fine-tuned personas rewrite linear truth readouts claim by claim, across four families and up to 32B.
- The model computes its answer largely at the answer position, from information that does not pass through those readouts, including a readout trained at the answer position itself.

**Does not show:**
- That the truth direction is irrelevant. It is a weak causal input, clearly beyond random directions.
- That this holds for every persona, for fine-tuning on unrelated data, or for emergent misalignment.
- Anything about non-linear readouts.

In the terms of the three outcomes, personas put linear truth probes in case C. A probe-based lie detector would not flag the persona's false answers, because the probe agrees with them, and an honesty intervention on the probed direction would barely change what the model says, because the answer is computed elsewhere.

For people building probes: a probe that tracks the belief you care about, even one that tracks the output at r = 0.9, may still be one the output ignores. Ablate it, steer it, and patch it before relying on it. In our setups, a faithful probe and an uninvolved probe were the same probe.

## Limitations

- Prompted personas plus two context-distilled LoRA personas; no personas learned from independent data.
- We or an LLM agent wrote the claim sets. Some "contested" partisan claims are well supported by evidence.
- Linear probes only. Logistic-regression and mean-difference probes disagree in places (section 8).
- Blocking the answer tail from seeing the prompt also removes the instruction, so the "statement route alone" test can't be read.
- Attention-blocking runs used eager attention, under which Qwen's baseline effect is 12–30% smaller (the other models agree).
- Partisan liberal-vs-conservative patching and the INLP layer-skipping fix were not pre-registered.
- Free-text verdicts come from a regex parser over 24 greedy tokens; under 2% of replies were unparseable.

## References

- Betley et al. (2025). Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs. [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)
- Chen, Arditi, Sleight, Evans & Lindsey (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)
- Kozlowski, Mintz & Lee (2026). Persona Perspectives: Adopting Viewpoints Reorganizes Representations in LLMs. [arXiv:2611.06848](https://arxiv.org/abs/2611.06848)
- Lampinen, Li, Hosseini, Bhardwaj & Shanahan (2026). Linear representations in language models can change dramatically over a conversation. [arXiv:2601.20834](https://arxiv.org/abs/2601.20834)
- Marks & Tegmark (2023). The Geometry of Truth. [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)
- Nadaf (2026). Emergent Misalignment Recruits a Pre-existing Persona Subspace. [arXiv:2607.21356](https://arxiv.org/abs/2607.21356)
- Ravfogel et al. (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. [arXiv:2004.07667](https://arxiv.org/abs/2004.07667)
- Sturgeon, Africa & Black (2026). When Role-playing, Do Models Believe What They Say? [arXiv:2606.11502](https://arxiv.org/abs/2606.11502); [LessWrong post](https://www.lesswrong.com/posts/EJQngix4rAgpPDTpT/when-role-playing-do-models-believe-what-they-say)
- Wang, Dupré la Tour, Watkins et al. (2025). Persona Features Control Emergent Misalignment. [arXiv:2506.19823](https://arxiv.org/abs/2506.19823)
- Zhong & Li (2026). Refusal Lives Downstream of Persona in Chat Models. [arXiv:2606.26161](https://arxiv.org/abs/2606.26161)

See also: von Klinski, Lapuschkin, Samek & Bürger (2026), Stress-Testing LLM Lie Detectors: Role-Play Failures and Spurious Correlations.

## Where each number comes from

Every result file logs the model, layer, α, vector norm, seed and probe accuracy. `{model}` is the model's Hugging Face name without the organization (e.g. `Qwen2.5-7B-Instruct`). The pre-registered rule for each experiment is in its script's docstring.

| Result | Section | Script | Result files in `results/` |
|---|---|---|---|
| Truth-probe shift, answer accuracy, 2020 control (Figure 1) | 1 | `experiments/e12_historical/gate.py` | `e12a_historical_gate_{model}.json` |
| Partisan and "opposite day" personas | 1 | `experiments/e12_historical/gate.py` | `e12a_partisan_gate_*.json`, `e12a_opposite*.json` |
| Statement and answer-tail patching (Figure 2) | 2 | `experiments/e12_historical/patch.py` | `e12c_historical_patch_{probe,ask,ask_suffix}_{model}.json`; partisan contrast: `e12c_partisan_*` |
| Attention blocking | 2, Limitations | `experiments/e16_answer_path/block.py` | `e16_block_{model}.json` |
| Truth-direction ablation (0–13%) | 3 | `experiments/e12_historical/ablate.py` | `e05_historical_ablation_{model}.json`, `e05_partisan_ablation_{model}.json` |
| Steering along the truth direction, INLP (Figure 3) | 3 | `experiments/e15_truth_causal/truth_causal.py` | `e15_truth_causal_{model}.json`; E20 reruns: `*_e20_statement.json`, `*_e20_answer.json` |
| Answer-position probe, probe types, component patching (Figure 4) | 4, 8 | `experiments/e19_answer_position/answer_pos.py` | `e19_answer_position_{model}.json` |
| Fine-tuned (LoRA) personas | 5 | `experiments/e18_lora_persona/distill.py`, then the e12 scripts with `--adapter(s)` | `e12a_historical_gate_lora_{model}.json`, `e12c_historical_patch_*_hist_{model}_*.json`, `e05_historical_ablation_lora_{model}.json` |
| 1850 physician LoRA | 5, 8 | same, `--setup physician1850` | `*physician1850*` |
| Free-text answers (Figure 5) | 6 | `experiments/e22_generation/gen.py` | `e22_generation_historical_{model}.json` |
| Persona vectors, mediated vs direct term | 7 | `experiments/e03_cross_layer/` | `e03_cross_layer_{model}.json` (Llama under `e03_llama/`) |
| Steer-then-knockout | 7 | `experiments/e14_knockout/knockout.py` | `e14_knockout_{model}*.json` |
| Emergent-misalignment organisms | grid | `experiments/e09_em_organisms/` | `e09*`, `e12c_historical_patch_*_{organism}_*.json` |
| All figures | | `post/make_figs.py` | reads the files above |

Run log and history of every experiment: [EXPERIMENTS.md](EXPERIMENTS.md).

*Thanks to … [acknowledgements].*
