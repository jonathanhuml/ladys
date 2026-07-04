# Third-Party Code

LaDyS includes an adapted subset of the CASSM filtering code inside
`src/ladys/models/_filtering_core.py`. The full upstream package is not copied
into this repository.

- Source package: `cassm`
- Version adapted from: `0.2.0`
- Source URL: https://pypi.org/project/cassm/0.2.0/
- License: MIT, see `CASSM-MIT.txt`

LaDyS includes a native PyTorch adapter inspired by the NeuralDataTransformer
(NDT) model and Lorenz/chaotic-RNN experiment configs. The upstream repository
is not vendored.

- Source package: `neural-data-transformers`
- Source URL: https://github.com/snel-repo/neural-data-transformers
- License: Unlicense, see `NEURAL-DATA-TRANSFORMERS-UNLICENSE.txt`

The `chaotic_rnn` dataset follows the random-RNN synthetic data recipe used by
LFADS/NDT examples, implemented here directly against the LaDyS dataset API.
