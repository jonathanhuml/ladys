# GPFA Port Notes

The local MATLAB reference is `/Users/jonathanhuml/Desktop/gpfa-matlab`.
The Python Elephant reference is:
`https://github.com/NeuralEnsemble/elephant/tree/master/elephant/gpfa`.

The implementation in `zynamics.models.gpfa` follows the MATLAB EM structure:

- `gpfaEngine.m`: initialize `gamma`, `eps`, `C`, `d`, and diagonal `R`.
- `fastfa.m`: initialize `C`, `d`, and `R` with factor analysis.
- `exactInferenceWithLL.m`: run the GPFA E-step and compute likelihood.
- `em.m`: update `C`, `d`, `R`, then learn GP kernel parameters.
- `learnGPparams.m` and `grad_betgam.m`: update each RBF `gamma` from
  posterior autocovariance sufficient statistics.

Elephant makes two translation choices that are useful but not copied exactly:

- It uses `sklearn.decomposition.FactorAnalysis` instead of the MATLAB `fastfa`
  EM loop. Zynamics ports the FA EM equations in torch to avoid adding sklearn
  to the model dependency surface.
- It uses SciPy `L-BFGS-B` on the same `grad_betgam` objective. Zynamics uses
  PyTorch autograd with `torch.optim.LBFGS` on the equivalent objective so the
  update can operate directly on tensors.

The benchmark epoch definition is unchanged: one GPFA epoch is one full E-step,
one C/d/R M-step, and the associated `gamma` optimization using the E-step
sufficient statistics.
