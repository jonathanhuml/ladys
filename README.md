# LaDyS

<p align="center">
  <img src="https://zkunkworks.com/ladys/assets/lorenz.png" alt="LaDyS Lorenz attractor logo" width="240">
</p>

LaDyS (Latent Dynamical Systems) is a Python library for fitting and comparing
models of neural population dynamics through a shared PyTorch API.

## Models

- bGPFA
- CASSM
- GPFA
- iLQR-VAE
- Kalman filter
- LangevinFlow
- LFADS
- MINT
- NDT
- PSTH
- Gaussian smoothing
- STNDT

## Quickstart

With LaDyS installed, run a Lorenz experiment on CPU:

```bash
ladys run -d lorenz -m gpfa --epochs 20 --device cpu
```

The experiment saves its configuration, fitted model, metrics, and predictions
under `runs/`.

## Learn more

- [Lorenz tutorial](tutorials/lorenz.ipynb): configure an experiment, train a model,
  and plot learning curves and reconstructed activity.
- [Documentation](https://zkunkworks.com/ladys/): model reference, configuration,
  and hyperparameter tuning.
