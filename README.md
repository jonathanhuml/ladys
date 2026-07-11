# LaDyS (Latent Dynamical Systems) Package

<p align="center">
  <img src="https://zkunkworks.com/ladys/assets/lorenz.png" alt="LaDyS Lorenz attractor logo" width="240">
</p>

PyTorch benchmark scaffolding for latent variable models of neural dynamics.

The first API pass standardizes model construction, training/reporting contracts,
and synthetic Lorenz/chaotic-RNN datasets. Models accept `(batch, time, neurons)`
tensors in `forward`.

## Initial Examples

- `ladys.models.bgpfa`: Bayesian GPFA using the vendored `mgplvm-pytorch`
  implementation with a differentiable variational ELBO and full-batch
  gradient strategy.
- `ladys.models.cassm`: thin adapter around the compact sparse CASSM
  filtering core in `ladys.models`.
- `ladys.models.gpfa`: Gaussian-observation GPFA with FA initialization and a
  differentiable exact marginal negative log likelihood. It trains with the
  standard PyTorch gradient strategy by default; the older EM adapter remains
  available through config.
- `ladys.models.ilqr_vae`: PyTorch iLQR-VAE adapter with posterior-control
  inference for pretrained MC_Maze checkpoints and ELBO training for new
  spike-count datasets.
- `ladys.models.kalman`: dense Kalman filter baseline adapted from the CASSM
  filtering code, exposed with per-trial rate predictions for benchmark metrics.
- `ladys.models.mint`: inference-only Mesh of Idealized Neural Trajectories
  decoder. MINT builds a trajectory library once, then decodes by Poisson
  likelihood recursion and interpolation; Lorenz defaults to spike-derived
  smoothed libraries, while `lorenz_library_source: true_rates` is reserved for
  oracle/debug checks.
- `ladys.models.ndt`: masked-count NeuralDataTransformer (NDT)
  adapter with native LaDyS config, training, prediction, and metrics contracts.


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
ladys run -c configs/experiment/synthetic/lorenz/bgpfa/bgpfa_lorenz.yaml
ladys run -c configs/experiment/synthetic/lorenz/gpfa/gpfa_lorenz.yaml
ladys run -c configs/experiment/synthetic/lorenz/ndt/ndt_lorenz.yaml
ladys list datasets
ladys list models
```

Synthetic and real-data tasks share the same CLI, but use task-specific
evaluation adapters under the hood. Synthetic datasets such as Lorenz and
chaotic RNN expose true rates and latents, so the default adapter reports
ground-truth rate and latent metrics. NLB datasets expose held-in spikes and
held-out targets, so the NLB co-smoothing adapter reports held-out bits/spike;
models can override `evaluation_adapter(task)` when they need a method-specific
readout, such as GPFA fitting a PyTorch linear or Poisson readout from inferred
features to held-out neurons.

## Neural Latents Benchmark Data

LaDyS can prepare the four core NLB'21 datasets, `area2_bump`, `mc_maze`,
`mc_rtt`, and `dmfc_rsg`, as held-in/held-out co-smoothing H5 files in
`data/real/nlb`. With `--download`, the command fetches the public NLB target
H5 and the required DANDI NWB files before building LaDyS-ready tensors:

```bash
PYTHONPATH=src python3 scripts/prepare_nlb_data.py \
  --datasets area2_bump mc_maze mc_rtt dmfc_rsg \
  --splits test \
  --bin-sizes-ms 5 20 \
  --download
```

If NWB files are already present in a DANDI-style directory, omit `--download`
and pass `--nwb-root` or repeated `--search-root` values. If the public target
H5 is already local, pass `--target-h5`. Dataset configs for the 5 ms and 20 ms
NLB test files live under `configs/dataset/`.

After a LaDyS run writes `predictions.npz`, score held-out count predictions
with the NLB co-smoothing bits/spike metric:

```bash
ladys run -c configs/experiment/real/mc_maze/ilqr_vae/ilqr_vae_mc_maze_nlb_5ms.yaml
ladys score-nlb --run-dir runs/ilqr_vae_mc_maze_nlb_5ms
```

GPFA uses the same real-data path with its NLB adapter:

```bash
ladys run -c configs/experiment/real/mc_maze/gpfa/gpfa_mc_maze_nlb_5ms.yaml
ladys score-nlb --run-dir runs/gpfa_mc_maze_nlb_5ms
```

For methods that emit a full EvalAI-style submission H5, the same command can
delegate to the `nlb_tools` evaluator:

```bash
ladys score-nlb --submission-h5 path/to/submission.h5 \
  --target-h5 data/real/nlb/eval_data_test.h5
```

## Scaling Benchmark

Install the plotting extra before running benchmark artifact scripts in a clean
environment:

```bash
pip install -e ".[benchmarks]"
```

Benchmark figures use a shared LaDyS plotting style backed by TUEplots when it
is available. On Python versions where TUEplots cannot be installed, LaDyS uses
a local Matplotlib fallback with the same figure sizing, color, grid, legend,
and export defaults.

```bash
PYTHONPATH=src python3 scripts/benchmark_lorenz_scaling.py \
  --models cassm gpfa kalman ndt \
  --neurons 10 100 1000 \
  --seeds 1
```

The script writes a grouped run under `runs/lorenz_scaling/`, including
`summary.csv`, `summary.npy`, `summary.md`, and
`plots/time_vs_neurons.png`.

## Loss-Curve Benchmark

```bash
PYTHONPATH=src python3 scripts/benchmark_lorenz_loss_curves.py \
  --models bgpfa cassm gpfa ilqr_vae kalman lfads mint ndt \
  --neurons 100 \
  --epochs 50
```

The script writes a grouped run under `runs/lorenz_loss_curves/`, including
top-level `summary.csv`/`summary.md`, comparison plots under `plots/`, and
per-model outputs under `models/<model>/`.

MINT appears in these curves as a horizontal inference-only baseline. Its
default Lorenz adapter estimates trajectory-library rates from smoothed training
spikes and repeated-trial averages. Passing
`--mint-lorenz-library-source true_rates` switches to an oracle sanity check and
should not be used for fair method ordering.

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

`configs/experiment/synthetic/lorenz/cassm/cassm_lorenz.yaml` and
`configs/experiment/synthetic/lorenz/kalman/kalman_lorenz.yaml` enable this
CASSM-style spike smoothing.
`configs/experiment/synthetic/lorenz/gpfa/gpfa_lorenz.yaml` leaves observations
raw.

See `website/docs/model_output_contract.md` for the forward-output convention.
See `website/docs/optimizer_contract.md` for the benchmark epoch definition.

## Documentation Site

The website workspace lives under `website/`. LaDyS docs source lives under
`website/docs/`, theme overrides live under `website/overrides/`, and MkDocs
builds generated output into `website/site/`. The standalone zkunkworks
homepage source lives under `website/home/`.

Model pages under `website/docs/models/` are generated from model class
docstrings and config defaults. Images referenced from model docstrings should
live under `website/docs/assets/` and can be linked as `assets/<filename>`:

```bash
python scripts/generate_model_docs.py
python scripts/generate_model_docs.py --check
```

Install the docs extra and serve the site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
