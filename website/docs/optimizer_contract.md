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

Library methods use `optimization.name: library_fit`. With MINT's
`train_source: lfads`, each epoch trains the rate estimator for one full pass
and updates the trajectory library. `trainer.epochs` controls the number of
these updates. Training history records MINT's training and validation Poisson
losses and the estimator's objective under `train_metrics.trajectory_loss`.
Resumable checkpoints include estimator weights and optimizer state; fitted
model checkpoints include the library needed for inference.

MINT's default `train_source: h5` and PSTH estimate a statistical library
directly. They use one full-data fitting epoch, calling
`model.fit_training_data(train_loader, device=device)`, followed by training
and validation loss measurement. Zero-epoch library training is rejected.
The model field `lfads_epochs` is retained for direct fitting helpers and
legacy reproduction runners; standard experiments use `trainer.epochs`.

The reporting contract is shared across strategies: every epoch returns a
`StepResult`, and benchmark plots use `seconds_per_epoch`. The benchmark
records training epoch time, including library learning and training-loss
measurement; validation and downstream metric computation are not included
in `seconds_per_epoch`.
