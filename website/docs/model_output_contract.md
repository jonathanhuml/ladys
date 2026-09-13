# Model Output Contract

All models must accept observations in `forward(x)` with shape:

```text
(batch, time, neurons)
```

The return value is `ModelOutput`. Some methods cannot cheaply populate every
field during the training path, so benchmark metrics should use
`model.predict_rates(x)` when they specifically need firing-rate curves. The
stable output fields are:

- `rates`: predictions in `(batch, time, neurons)` format.
- `rates_unit`: `"counts"` (expected spikes per input bin, the default) or
  `"hz"` (spikes per second). Poisson scoring consumes expected counts.
- `full_rates_unit`: required when `extras["full_rates"]` is present; that
  tensor may use different units from the primary held-out predictions.
- `latents`: inferred latent trajectories in `(batch, time, latent_dim)` format
  when the method exposes them.
- `reconstruction`: model reconstruction in observation space. For count models
  this may be rates; for Gaussian models this may be the conditional mean.
- `distribution`: optional PyTorch distribution or distribution-like object
  used when metrics need uncertainty or likelihood values.
- `extras`: method-specific diagnostics such as posterior variances, ELBO terms,
  marginal log likelihoods, or internal states.

Exporters use `output.count_rates(dt, full=True)`. A 20 Hz rate at 5 ms becomes
0.1 expected spikes; a prediction already equal to 0.1 counts stays 0.1.
Never multiply every model output by `dt` unconditionally. LFADS declares Hz;
NDT, STNDT, LangevinFlow, and MINT declare counts. iLQR-VAE's primary output is
counts while its full-neuron diagnostic output is Hz.

Synthetic evaluation normalizes predictions and declared dataset rate targets
to Hz for MSE/R2, and uses counts separately for Poisson metrics. By default
`predict_rates()` uses `ModelOutput.rates` and then `reconstruction`; methods
such as CASSM can override it to call their native prediction path.

Future benchmark tasks may add metrics that use `latents` for recovery of the
known Lorenz state, `distribution` for calibration/log-likelihood, and `extras`
for method-specific diagnostics. The forward signature should not change.

## Task Evaluation Adapters

Training-time `forward` outputs should stay model-native. Dataset/task-specific
evaluation lives in adapters selected by `model.evaluation_adapter(task)`:

- synthetic tasks use `SyntheticEvaluationAdapter`, which compares predicted
  rates/latents against dataset-provided ground truth.
- real NLB tasks use `NLBCoSmoothingAdapter`, which produces held-out expected
  spike counts for co-smoothing metrics. If a model directly returns held-out
  counts, as iLQR-VAE does, no extra readout is fitted. If it exposes latent
  features only, as GPFA does, the adapter can fit a PyTorch ridge or Poisson
  readout from training held-in features to training held-out targets.

This keeps a model's scientific output stable while allowing each benchmark
task to define the prediction target and metric surface it needs.
