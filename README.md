# zynamics

PyTorch benchmark scaffolding for latent variable models of neural dynamics.

The first API pass standardizes model construction, training/reporting contracts,
and a Lorenz synthetic dataset. Models accept `(batch, time, neurons)` tensors in
`forward`.

## Initial Examples

- `zynamics.models.cassm`: thin adapter around the upstream sparse CASSM
  `KalmanFilterSmoother` implementation. Install the original `cassm` package
  or the local CASSM repository before constructing this model.
- `zynamics.models.gpfa`: Gaussian-observation GPFA with FA initialization,
  EM updates, and RBF GP timescale learning. One full-dataset E/M update is
  treated as one benchmark epoch.

## References Inspected

- Local planning note: `/Users/jonathanhuml/Desktop/npdb.md`
- CASSM Lorenz data source: `jonathanhuml/cassm/src/cassm/datasets`
- Local GPFA-MATLAB reference: `/Users/jonathanhuml/Desktop/gpfa-matlab`

## Smoke Usage

```bash
PYTHONPATH=src python3 examples/smoke_compare.py
```

## Scaling Benchmark

```bash
PYTHONPATH=src python3 scripts/benchmark_lorenz_scaling.py \
  --models cassm gpfa \
  --neurons 10 100 1000 \
  --seeds 1
```

The script writes `lorenz_scaling_results.csv`,
`lorenz_scaling_results.npy`, and `time_vs_neurons.png` under
`artifacts/lorenz_scaling/`.

See `docs/model_output_contract.md` for the forward-output convention.
See `docs/optimizer_contract.md` for the benchmark epoch definition.
