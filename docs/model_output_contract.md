# Model Output Contract

All models must accept observations in `forward(x)` with shape:

```text
(batch, time, neurons)
```

The return value is `ModelOutput`. Some methods cannot cheaply populate every
field during the training path, so benchmark metrics should use
`model.predict_rates(x)` when they specifically need firing-rate curves. The
stable output fields are:

- `rates`: predicted firing-rate curves in `(batch, time, neurons)` format.
  This is the primary Lorenz benchmark output.
- `latents`: inferred latent trajectories in `(batch, time, latent_dim)` format
  when the method exposes them.
- `reconstruction`: model reconstruction in observation space. For count models
  this may be rates; for Gaussian models this may be the conditional mean.
- `distribution`: optional PyTorch distribution or distribution-like object
  used when metrics need uncertainty or likelihood values.
- `extras`: method-specific diagnostics such as posterior variances, ELBO terms,
  marginal log likelihoods, or internal states.

For the Lorenz task, the default accuracy metric should compare
`model.predict_rates(x)` against the generated ground-truth rates. By default
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
