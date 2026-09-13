# Repository Readiness Audit

Date: 2026-09-13. Baseline source revision: `23901de`.

## Implementation Update

The findings below describe the original audit, before the subsequent fixes.
Current core 5 ms coverage is **48/48 model/dataset combinations**, with
validation recipes for all 12 models on MC_Maze, Area2_Bump, MC_RTT, and DMFC_RSG.
Configuration coverage is not evidence of converged or published performance.

- Shared outputs now declare Hz versus expected counts, and both full and
  sharded NLB exporters convert exactly once. Synthetic rate errors use Hz;
  Poisson metrics and saved-artifact rescoring use expected counts.
- Missing training tensors raise errors. Explicit evaluation-only loading
  leaves training arrays absent instead of substituting evaluation data.
- Core 5 ms recipes use validation. STNDT/LangevinFlow checkpoint selection
  derives targets from that validation file and rejects external test targets.
  Standard fixed-budget test runs do not feed test losses to the scheduler.
- MINT fits through the standard trainer, serializes its fitted state, derives
  neuron layout from training data, and trains LFADS from raw spikes for RTT.
  Model-embedded paths/counts and Allen MINT's dated checkpoint dependencies
  were removed; explicit legacy reproduction options remain available.
- iLQR-VAE trains from random initialization, including MC_Maze. Student-prior
  Hessian scaling, terminal observation cost, line-search predictions, and
  differentiable fallback were corrected. Finite-difference tests validate
  derivatives of the executed finite-iteration solver, not upstream's implicit
  adjoint. Converged training and upstream numerical parity remain unverified.
- bGPFA now infers and decodes each actual input batch, freezes learned
  observation/GP parameters during inference, and restores lazy checkpoints.
- GPFA/Kalman canonical 5 ms recipes now perform gradient training. The
  original audit matrix overstated their zero-epoch inference recipes.
- Missing classical configs, stale config inventory tests, generated docs,
  and the misplaced Allen prediction-ensemble manifest were corrected.

MINT's newly trained LFADS pipeline is not AutoLFADS population-based tuning;
prepared DMFC condition averaging is not the legacy event-warped adapter.
Historical scores must not be attributed to these revised recipes. The 20 ms
matrix and legacy Allen iLQR checkpoint-transfer recipes are outside this
5 ms-focused pass. CASSM's separate output-directory ownership concern below
also remains outside the model/evaluation fixes.

### Verification After Fixes

- Local full suite: **205 passed, 4 CUDA-only tests skipped**.
- HAL full suite: **209 passed**, including CUDA training and checkpoint/device
  checks. Dependency deprecation warnings remain; no test failures.
- Real-data GPU diagnostics: **12/12 passed** for MINT, iLQR-VAE, and bGPFA
  across all four core datasets. Each uses 4 training trials, 2 validation
  trials, 12 bins at 5 ms, and small model dimensions. Gradient methods run
  two epochs; RTT MINT trains LFADS for one epoch before fitting its library.
  Losses/scores are finite, checkpoints restore equivalent predictions, and
  MINT/iLQR direct and exported held-out counts agree.
- DMFC's early bins are intentionally unscored. The diagnostic selects the
  first fully scored window using only target-mask finiteness, never spike
  values or achieved scores. This selection has a regression test.
- Prepared genuine validation data from the four training NWBs on HAL, with
  condition/PSTH metadata where supported. Fixed an unordered NWB clock before
  upstream positional resampling; irregular clocks now fail explicitly.
  `pandas<3` is required by the current `nlb-tools` mutation behavior.
- Generated documentation and `git diff --check` pass.

The diagnostic report is [verification.json](runs/nlb_readiness_20260913/verification.json).
Prepared validation files are available locally in `data/real/nlb/`. HAL used
an isolated checkout at `/tmp/ladys-readiness-20260913`, with dependencies
installed into that checkout's `test-deps`; the existing GPU environment and
remote repository were not overwritten. The repeatable diagnostic is
`scripts/verify_nlb_models.py`; `scripts/hal.sh` provides a fixed SSH entry point.

These are correctness and lifecycle checks, **not converged benchmarks**.

## Synthetic Follow-Up

This pass read every registered method implementation, including its training,
prediction, and evaluation paths, and added small Lorenz **and chaotic-RNN**
checks. Method-level runtime checks are not a guarantee of scientific parity.

### Additional Corrections

| Method | Confirmed corrections |
| --- | --- |
| MINT | Absolute Newton-step convergence; single-state interpolation; raw-count fitting; unsmoothed bin aggregation; nested-Subset condition mapping; dataset-derived exposure in the synthetic runner. |
| iLQR-VAE | Floating predictions for integer/bool spikes; raw-count posterior/ELBO/initialization; exposure and dimension validation; rejection of ELBOs missing parameter-dependent normalization. |
| bGPFA | Low-neuron factor-analysis initialization preserves requested latent rank and learnable scales; reject one-bin sequences; benchmark preserves full-batch trial identity. |
| GPFA | Initialization, EM, and prediction follow model dtype/device. |
| Kalman | Correct Matern process noise, verified by stationary covariance identity and gradients. |
| CASSM | Correct Gaussian KL trace sign and dimension term; use same-time posterior residual/covariance in the expected likelihood; reject sparse projections that would drop neurons. Full-projection losses and gradients match exact Gaussian references. |
| NDT | Correct synthetic/NLB readout sizing; raw/reconstruction targets; finite-target masking; no visible-input fallback when masked targets are missing; dtype compatibility. |
| STNDT | Correct synthetic/NLB readout sizing, raw/reconstruction targets, float64 embedding paths, and missing-target validation. |
| LFADS | Preserve raw Poisson targets with configurable recognition inputs; reject entirely missing targets. Dataset-derived exposure and empirical training bias are used in the benchmark. |
| LangevinFlow | Correct transition KL mean/log-variance arguments; synthetic/NLB readout sizing; raw/reconstruction targets; input precision and target validation; explicit paper-current versus released-code-lagged encoder alignment. |
| PSTH | Fit training state through the trainer; portable checkpoint buffers; no evaluation-batch fitting or missing-training fallback. |
| Smoothing | Raw-count synthetic evaluation; shared preprocessing preserves sequence length even when the kernel is longer than the trial. |

Lorenz now rejects invalid or empty splits and nonpositive/nonfinite numerical
parameters. Chaotic RNN rejects nonfinite dynamics/exposure parameters. Tests
cover both generators' rates/counts units and independent, balanced spike
repeats. Neither generator currently holds out entire condition trajectories.

The shared small-run driver now constructs models from dataset metadata, saves
completed epochs atomically, isolates the metric RNG, and distinguishes MINT
library growth from optimizer epochs and fixed baselines. The supervisor adds
a hard per-method timeout; a timeout is not reported as successful completion.

CASSM/Kalman's legacy model-local `save_model=True` only created hard-coded
directories without writing checkpoints. It now raises an actionable error;
`Experiment.run()` owns actual model checkpoint saving under `output_dir`.

LangevinFlow's default `encoder_input_alignment: current` follows
[paper Algorithm 1, line 9](https://arxiv.org/html/2507.11531v2).
The [released code](https://github.com/KingJamesSong/LangevinFlow_CCN/blob/main/nlb_lightning/models.py)
instead feeds the first bin twice and does not consume the last observed bin;
that behavior is available explicitly as `upstream_lagged`. Both modes are
tested. Final diagnostic curves use the current-bin mode; the earlier lagged
fits are archived separately. The corrected KL also intentionally differs
from the released code's variance/log-variance arguments.

Final verification: **424 passed, 6 skipped locally; 429 passed, 1 skipped on
HAL**, including CUDA checks. All 24 method/dataset jobs completed, with 40
epochs for the nine iterative methods, four MINT library fits, and one point
for each fixed baseline. The slowest fit took about 109 seconds. These counts
include the full repository suite, not only tests added in this audit.

### Remaining Priorities

1. Establish performance, not only runtime correctness. These small single-seed
   runs are diagnostics; 40 epochs can leave optimizer/regularization warmups
   unfinished. Revisit methods that do not beat smoothing before scaling up.
   In particular, the paper-current LangevinFlow chaotic-RNN run worsens from
   188 to 628 Hz squared despite improving count-reconstruction loss. GPFA and
   Kalman also worsen on chaotic RNN, and LFADS' best epoch precedes its final
   epoch. These are failed quality checks, even though execution is finite.
2. Consolidate synthetic recipes. Chaotic RNN currently has only an NDT
   experiment YAML; other methods run from explicit generated configurations
   or generic defaults. Lorenz also lacks a canonical STNDT YAML. Save tested
   dataset-aware recipes, with suitable Gaussian preprocessing, rather than
   treating config presence as evidence of tuned performance.
3. Finish iLQR's real-data convergence work and upstream comparison. Its
   finite unrolled solver is not the upstream implicit adjoint; prediction
   currently decodes posterior-mean controls rather than averaging covariance.
4. Revalidate historical results affected by objective, MINT interpolation,
   masking, or unit changes. The earlier full-data NLB scores are artifacts of
   their recorded snapshot, not scores for every subsequent source revision.
5. Make bGPFA ELBO diagnostics independent of evaluation minibatch partition
   and align their reported annealing schedule with the optimizer. Rate-error
   checks avoid comparing these non-equivalent native ELBO values.
6. Document intentional upstream architecture choices separately from bugs.
   STNDT uses spatial attention weights without its computed spatial value
   output, leaving spatial output-projection parameters unused. Altering that
   requires an explicit method variant or upstream parity investigation.
7. Add multi-seed checks and held-out-condition experiments. The present
   independent-spike-repeat split tests denoising familiar trajectories, not
   generalization to unseen dynamics. Keep 20 ms NLB secondary to 5 ms work.

### Reproduction

The bounded runner is `scripts/run_synthetic_readiness.py`. On HAL the stable
entry is `scripts/hal_synthetic.sh run <python> --device cuda --output-dir <run>`
through the approved `ssh -o BatchMode=yes -o ConnectTimeout=10 HAL` prefix.
It runs 12 neurons, 3 conditions, 6 trials per condition, 60 time bins, up to
40 epochs, and a 600-second hard cap per method/dataset. No host-specific data
or environment paths are embedded in these scripts. The results and plots are
in [the synthetic report](runs/synthetic_readiness_20260913/report.md).

## Original Audit

The repository contains substantial implementations and working training code, but it is not yet a consistent, portable benchmark for fitting every method from scratch. MINT, iLQR-VAE, and bGPFA require particular attention. Shared evaluation bugs also affect models whose NLB config coverage looks complete.

This audit inspected all 12 registered model presets, all 111 experiment YAMLs, the shared training/evaluation paths, local NLB H5 schemas, selected existing run artifacts, and upstream MINT/iLQR-VAE sources. It ran the existing test suite and small CPU reproductions of the issues below. It did not rerun full NLB training, inspect HAL, or independently reproduce published benchmark scores. Implementation and configuration files were not changed.

## Findings

### 1. P1: Shared NLB export applies the wrong rate units

In [nlb_eval.py](src/ladys/nlb_eval.py), line 282, `_collect_full_rate_parts` multiplies every model's rates by `dt`. NDT, STNDT, and LangevinFlow already return expected counts per bin: their Poisson losses compare these predictions directly with spike counts. The shared exporter therefore scales their predictions a second time.

Small runtime probes confirmed an export/direct prediction ratio of approximately `0.005` for all three models at 5 ms: exported counts are 200 times too small. This affects the standard `Experiment` full-NLB export and checkpoint evaluation using that exporter. The separate `run_nlb_classical_table.py` evaluator uses model adapters, so this finding alone does not establish that historical results from that script are wrong.

LFADS outputs Hz, and iLQR-VAE supplies Hz in `extras['full_rates']`; those cases explain why this multiplication exists. The fix needs an explicit, consistent unit contract and model-specific conversion where necessary, with direct/export agreement checked for every model.

### 2. P1: Missing training tensors silently turn evaluation data into training data

[datasets/nlb.py](src/ladys/datasets/nlb.py), lines 184-185, substitutes `eval_spikes_heldin` and `eval_spikes_heldout` when the corresponding training keys are absent. A minimal evaluation-only H5 reproduced both substitutions. A training run can consequently fit directly to its evaluation targets without an error.

The four local prepared core 5 ms files do contain separate training tensors, so this audit did not find that particular fallback active in those files. It remains a serious failure mode for incomplete or externally supplied H5s. Training should require explicit training data; evaluation-only loading should be a separate supported operation.

### 3. P1: Test labels are used for checkpoint selection

All 45 checked-in experiment YAMLs for the four core NLB datasets specify `split: test`; none supplies a canonical validation configuration. The LangevinFlow reproduction runner defaults to test data/targets and retains the checkpoint with the highest evaluated co-bps. STNDT's runner also selects checkpoints by those metrics when used with the checked-in test configs. Both support stopping based on evaluation performance.

Evidence: [run_langevin_flow_nlb_reproduction.py](scripts/run_langevin_flow_nlb_reproduction.py), lines 204, 327, and 366; [run_stndt_nlb_reproduction.py](scripts/run_stndt_nlb_reproduction.py), line 417. These results must be distinguished from performance on an untouched test set. Model and hyperparameter selection need a validation split, followed by a final test evaluation.

### 4. P1: MINT fitting is not integrated into the standard API

[cli.py](src/ladys/cli.py), line 147, sends every MINT configuration to `run_mint_nlb`, including synthetic and Allen VCN configurations. Running the checked-in Lorenz MINT YAML through the CLI reproduced `KeyError: 'lorenz'`.

Calling `Experiment.run()` directly avoids that dispatch but does not fit a synthetic MINT library: the inference-only strategy performs no fitting, and the synthetic evaluator has no MINT fitting hook. A small direct-API run reproduced `RuntimeError: MINT must be fit with fit_library() before predict().`

The library-fitting implementation itself exists at [models/mint.py](src/ladys/models/mint.py), line 405. Benchmark-specific scripts explicitly call it. MINT needs a normal fit lifecycle in the public experiment API, with the CLI respecting the dataset type.

### 5. P1: MINT does not offer a complete from-scratch pipeline for MC_RTT

The phrase "inference-only" understates what MINT does. Upstream MINT explicitly implements `fit`, which learns trajectory templates before decoding. This is a statistical fitting procedure; adding gradient descent merely to make it resemble a neural network would change the method. See upstream [fit.m](https://github.com/seanmperkins/mint/blob/main/core/%40MINT/fit.m).

The local Area2 and MC_Maze builders smooth and average training spikes; the main DMFC config fits from NWB trials. Those are real data-derived fits. However, the MC_RTT config uses MATLAB data containing precomputed AutoLFADS rates, and its trajectory builder takes rates from `Z[4:]`. Direct MC_RTT NWB training is explicitly rejected. The train-LFADS-then-fit-MINT implementation accepts only DMFC. Upstream also uses AutoLFADS-derived MC_RTT trajectories, so reproducing that full pipeline from raw data requires training that upstream component too. See the [upstream MINT description](https://github.com/seanmperkins/mint#task-related).

Evidence: [mint_nlb.py](src/ladys/mint_nlb.py), lines 219-248; [models/mint.py](src/ladys/models/mint.py), line 1859; [MC_RTT config](configs/experiment/real/mc_rtt/mint/mint_mc_rtt_nlb_5ms.yaml). All four selected Allen MINT configs additionally reference existing LFADS run directories. They are dependent experiment pipelines, not independent fits from raw spikes.

### 6. P1: Fitted MINT state is missing from checkpoints

MINT stores learned libraries and lookup/index tensors in ordinary attributes. Its only registered buffer is `_device_anchor`: [models/mint.py](src/ladys/models/mint.py), lines 325-336. A fitted model's `state_dict()` contained only that empty buffer. Loading it into a fresh instance succeeded, but prediction failed because the library was absent.

The dedicated NLB runner also writes predictions and metrics without saving the fitted model: [mint_nlb.py](src/ladys/mint_nlb.py), line 735. MINT needs serialization for its complete fitted state and a save/load prediction-equivalence check. Device movement after fitting needs to cover those tensors as well.

### 7. P1: bGPFA confuses different evaluation batches of the same size

[models/bgpfa.py](src/ladys/models/bgpfa.py), lines 250-271 and 320, keys the posterior cache only by the number of trials. The NLB adapter first infers latents for every evaluation batch, then makes another pass to decode them. Later batches overwrite earlier ones of the same size. If an evaluation batch has the same trial count as the full training set, `_model_for` returns the training posterior instead.

Runtime reproduction with two distinct, nondegenerate batches confirmed finite latents, replacement of the first batch's posterior after inference on the second, and identical posterior latents returned for both batches afterward. The inherited decoder-fitting loop also calls `model(x)` on training minibatches without selecting the corresponding fitted full-batch posterior rows.

bGPFA needs trial-aware posterior access or inference and decoding within the same batch operation. A missing-config fix alone would not make it NLB-ready.

### 8. P1: None of the four core NLB iLQR-VAE configs trains from scratch

The [MC_Maze config](configs/experiment/real/mc_maze/ilqr_vae/ilqr_vae_mc_maze_nlb_5ms.yaml) loads `final_params.bin`, freezes parameters, and runs zero training epochs. The Area2, MC_RTT, and DMFC files named `*_train.yaml` all set `initialization: checkpoint_transfer` and load the same tutorial checkpoint through `template_params_path`. The Allen iLQR-VAE configs also use checkpoint transfer.

There is working random-initialization code. A small CPU run with `initialization='random'`, no checkpoint, and an ELBO optimizer produced a finite loss, changed all 15 parameter tensors, and returned the expected held-out shape. The synthetic Lorenz preset and sweep generator expose this capability. The gap is a supported, validated set of from-scratch NLB recipes, not a wholly absent optimizer or solver.

Evidence: [models/ilqr_vae.py](src/ladys/models/ilqr_vae.py), lines 252-273; [Area2 config](configs/experiment/real/area2_bump/ilqr_vae/ilqr_vae_area2_bump_nlb_5ms_train.yaml), lines 12-13; [generate_ilqr_vae_sweep_configs.py](scripts/generate_ilqr_vae_sweep_configs.py), lines 49-51 and 78-81.

### 9. P1: The default iLQR-VAE training gradient differs from the original method

All three core NLB training configs set `differentiate_controls: false`. [models/ilqr_vae.py](src/ladys/models/ilqr_vae.py), line 420, explicitly detaches inferred controls from the ELBO gradient. The original method differentiates through the posterior-control solution; this is central to its recognition model. See the [paper](https://openreview.net/pdf?id=wRODLDHaAiW) and [upstream variational implementation](https://github.com/marineschimel/ilqr_vae/blob/main/lib/variational.ml).

The local opt-in differentiable path unrolls iLQR updates, as its docstring states; it is not the upstream implicit adjoint. Solver fallback also forces `differentiable=False`. These can be useful approximations, but the current configs should not be presented as validated training reproductions. Numerical gradient checks and training comparisons against upstream are missing from the checked-in tests. A test that verifies only `requires_grad` is insufficient for that claim.

### 10. P2: bGPFA checkpoint loading and synthetic evaluation are incomplete

The `_train_mod` module is created lazily when training data first arrives. A freshly built bGPFA therefore cannot load a populated training `state_dict` with the standard checkpoint evaluator. A small build/save/load reproduction raised a state-dict loading error. See [models/bgpfa.py](src/ladys/models/bgpfa.py), line 276, and [evaluate_nlb_checkpoint.py](scripts/evaluate_nlb_checkpoint.py), lines 38-40.

The generic synthetic evaluator also calls `model(x)` without first inferring new-trial latents. The dedicated synthetic benchmark script explicitly performs that inference. Standard `Experiment` and benchmark-script evaluations therefore do not currently have equivalent bGPFA behavior. See [metrics.py](src/ladys/metrics.py), line 73, and [benchmark_lorenz_loss_curves.py](scripts/benchmark_lorenz_loss_curves.py), line 917.

### 11. P2: Dataset assumptions and artifact paths are embedded in model code and generic presets

Concrete examples:

- [models/mint.py](src/ladys/models/mint.py), lines 29-41: fixed held-out neuron counts and dataset-to-NWB filename mappings.
- [models/mint.py](src/ladys/models/mint.py), line 1000: dataset-specific alignment windows, sampling intervals, smoothing dimensions, and other settings in Python branches.
- [mint_nlb.py](src/ladys/mint_nlb.py), lines 206-215 and 628-630: fixed 5 ms/300-bin assumptions for H5-backed DMFC and target selection using only the dataset key, despite exposing an evaluation bin-size setting.
- [models/ilqr_vae.py](src/ladys/models/ilqr_vae.py), line 43: a tutorial checkpoint path as the default model input.
- [configs/model/ilqr_vae.yaml](configs/model/ilqr_vae.yaml), lines 3 and 8-10: a supposedly generic preset contains a checkpoint path and MC_Maze-specific `137/45` neuron slices.
- [configs/model/mint.yaml](configs/model/mint.yaml): the generic model preset is tied to MC_Maze, NWB/MAT roots, and a public test target file.
- [models/_filtering_core.py](src/ladys/models/_filtering_core.py), lines 269 and 478: optional internal saving chooses `cassm_runs`/`kalman_runs` independently of the experiment output directory.
- Selected Allen MINT and ensemble configs point into dated `runs/...` artifact trees.

The core source scan did not find literal personal `/home/...` or `/Users/...` paths in the model files. The problem is primarily fixed relative artifact layouts, task assumptions, and duplicated metadata. Relative dataset paths supplied in experiment YAMLs are already overridable; they differ from filenames and neuron counts selected internally by a model.

Data paths and dataset metadata should come from the data layer; task presets should live in configs; output locations should come from the experiment. Models should infer neuron slices and timing from data where possible. Algorithm constants and configurable hyperparameter defaults do not need to be eliminated indiscriminately.

### 12. P2: Config validation and documentation checks are failing

Of 111 experiment YAMLs, 110 loaded through `load_experiment_config`. The [Allen CASSM ensemble YAML](configs/experiment/real/allen_vcn/cassm/cassm_allen_vcn_bo_drifting_one_tf_20ms_nlb_style_ensemble_top4.yaml) is a descriptor of existing prediction files, not an executable experiment config. Its string-valued `dataset` causes an `AttributeError` in the public loader. It needs an explicit manifest location/schema or a supported ensemble execution path.

The test suite returned **58 passed, 6 failed**:

| Failure | What it indicates |
| --- | --- |
| Experiment config loading | The Allen CASSM descriptor is not accepted by the standard loader. |
| iLQR-VAE config inventory | Test assumes only the four core configs exist; Allen configs violate that assumption. |
| LFADS config inventory | Same outdated inventory assumption. |
| STNDT config inventory | Same outdated inventory assumption. |
| MINT epoch trial counts | Function returns repetition counts `[1, 2]`; test expects trial counts `[3, 6]`. The API/test contract needs resolution. |
| Generated model docs | Eight pages are stale: bGPFA, CASSM, iLQR-VAE, LangevinFlow, LFADS, MINT, NDT, STNDT. |

The inventory failures are not evidence that LFADS or STNDT training itself is absent. Existing tests mostly check small model contracts, not complete raw-data preparation, fitting, serialization, and official NLB evaluation for every model/dataset combination.

## Core NLB Config Matrix

This is checked-in 5 ms experiment coverage, not a claim that training converged or published results were reproduced. `Train` means a gradient-training recipe exists; `Fit` means a statistical library/readout fitting recipe exists. Shared findings above still apply.

| Model | MC_Maze | Area2_Bump | MC_RTT | DMFC_RSG |
| --- | --- | --- | --- | --- |
| bGPFA | Missing | Missing | Missing | Missing |
| CASSM | Train | Train | Train | Train |
| GPFA | Zero-epoch inference | Missing | Missing | Zero-epoch inference |
| iLQR-VAE | Pretrained only | Transfer | Transfer | Transfer |
| Kalman | Zero-epoch inference | Missing | Missing | Zero-epoch inference |
| LangevinFlow | Train | Train | Train | Train |
| LFADS | Train | Train | Train | Train |
| MINT | Fit from NWB | Fit from NWB | Supplied AutoLFADS rates | Fit from NWB; optional LFADS fit |
| NDT | Train | Train | Train | Train |
| PSTH | Fit | Missing | Missing | Missing |
| Smoothing | Fit | Missing | Missing | Missing |
| STNDT | Train | Train | Train | Train |

There are **34 covered combinations out of 48**, leaving **14 missing**: bGPFA four; GPFA two; Kalman two; PSTH three; smoothing three. Duplicate/canonical variants produce 45 YAMLs across these four datasets; the file count should not be confused with coverage.

At **20 ms**, only LangevinFlow has all four core configs; GPFA and Kalman each have DMFC. That is **6 of 48 combinations**. The data layer supports MC_Maze large/medium/small, but no canonical experiment YAMLs were found for those three variants.

Missing configs do not always mean missing implementation. [run_nlb_classical_table.py](scripts/run_nlb_classical_table.py) constructs GPFA, Kalman, PSTH, and smoothing configurations programmatically for all four datasets. The local [20-epoch summary](runs/nlb_classical_20epoch/summary.csv) contains results for all 16 combinations. Those are historical artifacts, not fresh validations, but they support treating many classical-model config omissions as configuration consolidation work. bGPFA, in contrast, has missing core NLB configs plus the runtime issues above.

## Model Assessment

| Model group | Assessment |
| --- | --- |
| MINT | Real library fitting and decoding exist. Requires unified fit/save/load APIs, a raw-data MC_RTT pipeline, portable dataset inputs, and corrected CLI routing. |
| iLQR-VAE | Real random initialization, solver, ELBO, and parameter updates exist. NLB recipes depend on pretrained weights; gradient fidelity and converged from-scratch training remain unvalidated. |
| bGPFA | Real variational training exists. Posterior batching, checkpoint restoration, synthetic evaluation, and all four core NLB recipes need work. |
| LFADS | Substantial trainable implementation, all four 5 ms configs, and direct held-out adapter tests. Dataset-specific readout sizes remain in configs; this audit does not establish published-score parity. |
| NDT / STNDT / LangevinFlow | Substantial trainable implementations and all four 5 ms configs. Shared export units need fixing; validation/test selection needs separation in the reproduction workflows. |
| CASSM | Trainable filtering core, held-out readout, and all four 5 ms configs. Allen ensemble packaging and output-directory ownership need cleanup. |
| GPFA / Kalman | Trainable cores and NLB readouts exist. Missing canonical Area2/RTT configs despite programmatic runs. |
| PSTH / Smoothing | Fitted baselines are appropriate; an inference-only optimizer is not itself a missing-training bug. Six canonical dataset configs are absent. |

## Suggested Work Order

1. Fix benchmark correctness: explicit rate units, rejection of missing training tensors, validation/test separation, and bGPFA trial identity during inference.
2. Complete MINT's fit/save/load integration, move dataset plumbing out of the model, and provide MC_RTT fitting from raw spikes through a trained rate estimator.
3. Provide checkpoint-free iLQR-VAE NLB configs and verify its gradient against upstream, then establish converged training results.
4. Consolidate the working classical-model recipes into canonical configs and add the missing bGPFA recipes after its runtime issues are fixed.
5. Repair the six failing checks and add targeted integration checks for raw-data fitting, checkpoint restoration, and agreement between direct and exported NLB metrics.

## Verification Record

Existing suite: `env PYTHONPATH=src:. python3 -m pytest -q`, Python 3.9, PyTorch 2.8.0, CPU. Result: 58 passed, 6 failed in 10.64 seconds.

Additional small CPU probes confirmed:

- Evaluation-only H5 input and target arrays become training arrays without an error.
- Fitted MINT state dict contains only `_device_anchor`; restored prediction fails.
- The checked-in Lorenz MINT CLI config raises `KeyError: 'lorenz'`.
- Synthetic MINT through `Experiment.run()` reaches evaluation without a fitted library.
- Random iLQR-VAE training without a checkpoint updates 15/15 parameter tensors with finite loss and correctly shaped held-out predictions; default controls are detached.
- NDT, STNDT, and LangevinFlow shared exports are `0.005` times direct counts at 5 ms.
- A fresh bGPFA cannot directly restore a populated training state dict.
- Two different bGPFA evaluation batches of equal size reuse the last inferred posterior; the reproduced latents were finite.

These probes establish specific runtime failures and small-case trainability. They do not establish convergence, GPU behavior, numerical parity with every upstream implementation, or the validity of every historical benchmark artifact.
