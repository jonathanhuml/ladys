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

Library methods use `optimization.name: library_fit` and `trainer.epochs: 1`.
Their training epoch calls `model.fit_training_data(train_loader, device=device)`
to learn the complete library, then records training and validation losses.
MINT learns trajectory templates in this epoch, including optional LFADS rate
estimation. PSTH learns its training-trial average. A larger epoch budget still
completes after this full-data fit; zero epochs are rejected. Fitted state
survives checkpoint restoration and device moves, and resuming a completed
library fit does not retrain it.

The reporting contract is shared across strategies: every epoch returns a
`StepResult`, and benchmark plots use `seconds_per_epoch`. The benchmark
records training epoch time, including library learning and training-loss
measurement; validation and downstream metric computation are not included
in `seconds_per_epoch`.
