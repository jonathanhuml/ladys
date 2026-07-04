# Optimizer Contract

The trainer owns the benchmark epoch. An epoch means one full pass over the
training dataset, regardless of the optimizer family.

For gradient methods such as CASSM, `GradientStrategy.train_epoch` iterates the
`DataLoader` once. Each minibatch runs `forward`, backpropagation, and one
optimizer update.

For EM methods, `EMStrategy.train_epoch` iterates the `DataLoader` once to
assemble the observations for the method, then calls `model.fit_em_epoch(x)`.
That method should perform one full E-step/M-step update using that full pass.
Inner linear solves or hyperparameter optimizers inside the M-step are counted
inside the same epoch if they reuse the sufficient statistics from that pass. A
second full E-step over the dataset is a second benchmark epoch.

GPFA defaults to `GradientStrategy`: each minibatch computes the exact marginal
negative log likelihood and runs one standard PyTorch backward/optimizer step.
The older full-dataset EM adapter remains available by setting
`optimization.name: em`.

The reporting contract is shared across strategies: every epoch returns a
`StepResult`, and benchmark plots use `seconds_per_epoch`. The benchmark
records optimizer epoch time only; validation and downstream metric computation
are not included in `seconds_per_epoch`.
