[![install-test](https://github.com/microsoft/aoc/actions/workflows/install-test.yaml/badge.svg?branch=main)](https://github.com/microsoft/aoc/actions/workflows/install-test.yaml)
[![unit-test](https://github.com/microsoft/aoc/actions/workflows/unit-test.yaml/badge.svg?branch=main)](https://github.com/microsoft/aoc/actions/workflows/unit-test.yaml)

# Analog-Optical Computer (AOC)

This repo contains the code for the Digital Twin (DT) used in Microsoft Research's AOC project.

Paper: $PAPER

The DT provided in this repo can be used to train fixed-point models to run on either the newer 2300-weight or the 256-weight AOC machine that is published in the journal paper.

At its core, AOC implements the following idealized iteration

`z_t+1 = alpha z_t + beta W tanh(z_t) + x + b`,

which allows it to rapidly evaluate fixed-point models such as Deep-Equilibrium (DEQ) or Hopfield models. The input to the model `x` is reinjected at every iteration in line with the formulation of these models.
The output is then defined as the fixed-point `z*`.

However, various non-idealities perturb the computation in the AOC device away from the equation above, creating the need for a faithful digital twin through which the models can be trained digitally so that they can be evaluated correctly on AOC. 


## Setup
The package can be installed via `pip install aoc`. It requires python>=3.9 and a torch>=2.0 as well as torchdeq, 
einops, pyyaml, and pytest. Consult the pyproject.toml for details.

## Functionality
The core of the digital twin is the `AOCCell` class which attempts to model the physics of the analog-optical device.
`AOCCell` needs to be wrapped by initial and final layers ensuring correct input and output dimensionality.
We provide a ready-made wrapper that offers linear input and output projections with the `DEQInputOutputProjection` class.
We recommend instantiating this class via its `DEQInputOutputProjection.create_default_aoc_model(...)` factory method.

`AOCCell` supports a variety of structured matrices. The simplest matrix structure is dense and can be achieved by passing 
`d_hidden=[s, s]` where `s` is the desired size of the matrix. If a multi-layer structure should be simulated, more
intermediate sizes can be provided, for example, `d_hidden=[24, 24, 24]` creates a two-layer structure which would still 
fit the 2300-weight machine.

By default, a DEQ-structure is created which means that all layers are laid out below the diagonal with a single
feedback block on the top right. For the `d_hidden=[24, 24, 24]` configuration, this would mean that the matrix consists
of two `24x24`-shaped blocks along the antidiagonal.

The `MatrixConnectivityType` argument controls the type of the matrix and can be set to 
1. `Feedback` (i.e. DEQ).
2. `Feedforward`, which means that no feedback occurs through the matrix.
3. `Hopfield`, which ensures a symmetric matrix.

We also provide a more idealized implementation of the hardware operation in the `SimpleCell` class.
We recommend the constructor wrapper-model factory method `DEQInputOutputProjection.create_simple_cell_model(...)`
to experiment with this cell type.

## Hardware Options

The exact hardware configuration is controlled by the abstract `HardwareParameters` class which comes with two child classes: 
`HardwareParameters16` and `HardwareParameters48` corresponding to the previous and the current AOC hardware generation.
We provide a yaml configuration for each hardware generation in `src/aoc/hw_config`.

`AOCCell` itself offers further options to turn on or off a range of non-idealities. Each non-ideality is attempts to 
empirically model a particular effect in the hardware that is relevant enough to impact the result of the analog computation.
For details on these effects, please consult the paper supplementary section D.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
