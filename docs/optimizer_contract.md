# Optimizer Contract

The trainer owns the benchmark epoch. An epoch means one full pass over the
training dataset, regardless of the optimizer family.

For gradient methods such as CASSM, `GradientStrategy.train_epoch` iterates the
`DataLoader` once. Each minibatch runs `forward`, backpropagation, and one
optimizer update.

For EM methods such as GPFA, `EMStrategy.train_epoch` iterates the `DataLoader`
once to assemble the observations for the method, then calls
`model.fit_em_epoch(x)`. That method should perform one full E-step/M-step
update using that full pass. Inner linear solves or hyperparameter optimizers
inside the M-step are counted inside the same epoch if they reuse the sufficient
statistics from that pass. A second full E-step over the dataset is a second
benchmark epoch.

For GPFA specifically, RBF GP timescale learning is part of `fit_em_epoch`.
It uses posterior autocovariance sufficient statistics from the current E-step,
so the L-BFGS iterations over `gamma` are included in the same optimizer epoch.

The reporting contract is shared across strategies: every epoch returns a
`StepResult`, and benchmark plots use `seconds_per_epoch`. The benchmark
records optimizer epoch time only; validation and downstream metric computation
are not included in `seconds_per_epoch`.
