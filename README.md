# LaDyS (Latent Dynamical Systems) Package

<p align="center">
  <img src="https://zkunkworks.com/ladys/assets/lorenz.png" alt="LaDyS Lorenz attractor logo" width="240">
</p>

PyTorch benchmark scaffolding for latent variable models of neural dynamics.

The first API pass standardizes model construction, training/reporting contracts,
and a Lorenz synthetic dataset. Models accept `(batch, time, neurons)` tensors in
`forward`.

## Initial Examples

- `ladys.models.cassm`: thin adapter around the bundled sparse CASSM
  implementation shipped in `src/cassm`.
- `ladys.models.gpfa`: Gaussian-observation GPFA with FA initialization,
  EM updates, and RBF GP timescale learning. One full-dataset E/M update is
  treated as one benchmark epoch.


## Public Experiment API

LaDyS is organized around three public pieces:

- `ladys.models`: model configs build PyTorch `nn.Module` instances and carry
  their optimization settings.
- `ladys.data.DataModule`: creates PyTorch train/validation datasets and
  dataloaders.
- `ladys.Experiment`: gathers data, model, training, evaluation metrics, and
  artifacts into one inspectable run folder.

Run the canonical CLI path with a dataset and model:

```bash
ladys run -d lorenz -m cassm
```

The run folder includes `config.json`, `history.csv`, `metrics.json`,
`predictions.npz`, `model.pt`, and `report.md`. The CLI also accepts full YAML
experiment configs:

```bash
ladys run -c configs/experiment/gpfa_lorenz.yaml
ladys list datasets
ladys list models
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

## Loss-Curve Benchmark

```bash
PYTHONPATH=src python3 scripts/benchmark_lorenz_loss_curves.py \
  --models cassm gpfa \
  --neurons 100 \
  --epochs 30
```

The script writes `lorenz_loss_history.csv`, `test_rate_mse_curves.png`,
`test_objective_curves.png`, `train_test_objective_curves.png`, and per-model
held-out rate trace plots/CSVs under `artifacts/lorenz_loss_curves/`.

## Preprocessing

Experiment YAML files can include a `preprocessing` block. The benchmark
scripts apply this to dataset observations before models see them, while
leaving Lorenz ground-truth rates unchanged for MSE metrics.

```yaml
preprocessing:
  observations:
    name: smooth_firing_rate
    sampling_precision: 20.0
    kern_sd_ms: 50.0
```

`configs/experiment/cassm_lorenz.yaml` enables this CASSM-style spike
smoothing. `configs/experiment/gpfa_lorenz.yaml` leaves observations raw.

See `docs/model_output_contract.md` for the forward-output convention.
See `docs/optimizer_contract.md` for the benchmark epoch definition.

## Documentation Site

Docs source lives under `docs/` and is configured by `mkdocs.yml`. Model pages
under `docs/models/` are generated from model class docstrings and config
defaults. Images referenced from model docstrings should live under
`docs/assets/` and can be linked as `assets/<filename>`:

```bash
python scripts/generate_model_docs.py
python scripts/generate_model_docs.py --check
```

Install the docs extra and serve the site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
