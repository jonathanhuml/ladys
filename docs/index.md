# LaDyS

LaDyS (Latent Dynamical Systems) provides PyTorch benchmark scaffolding for
latent variable models of neural dynamics.

## Documentation

- [Model documentation](models/index.md)
- [Model output contract](model_output_contract.md)
- [Optimizer contract](optimizer_contract.md)

## CLI

Run a full LaDyS experiment with:

```bash
ladys run -d lorenz -m cassm
```

The package also includes `ndt` and `chaotic_rnn` entries in the public
model/dataset registries.

This builds the dataset, model, trainer, evaluation metrics, and a self-contained
run folder through the public `ladys.Experiment` orchestration API.

Model pages are generated from model class docstrings and config defaults:

```bash
python scripts/generate_model_docs.py
```

Images referenced from model docstrings should live under `docs/assets/`.
Use links like `![Diagram](assets/model-diagram.png)` in docstrings; generated
model pages rewrite those links to the correct relative path.

Serve the documentation site locally with:

```bash
mkdocs serve
```
