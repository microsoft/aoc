# Examples

Training examples for AOC DEQ models on classification and regression tasks.

## Cell Types

The examples support two cell types for the DEQ recurrent block:

- **SimpleCell** (default): Standard DEQ cell without hardware non-idealities. Use this for arbitrary hidden sizes and number of layers.
- **AOCCell** (`--use-aoc-cell`): Hardware digital twin that simulates real AOC hardware non-idealities (weight distortion, crosstalk, etc.). Use this to predict how models will perform on actual hardware.

## Running the Examples

All scripts support several command-line options to configure training:
- `--epochs` - number of training epochs
- `--hidden-size` - size of hidden layers
- `--num-layers` - number of DEQ layers (default: 1)
- `--use-aoc-cell` - use AOCCell (hardware digital twin) instead of SimpleCell
- `--fixed-point-init` - initialization for fixed-point solver: `zeros`, `x_proj`, or `random`

### AOCCell Hardware Constraints

When using `--use-aoc-cell`, the model uses hardware calibration data from actual hardware. This calibration data is only available for specific configurations:

- **Hidden sizes**: 16 or 48 only
- **Number of layers**: 1 only (i.e., `layer_sizes=[N, N]` internally)

Using other configurations will raise a `ValueError`. For arbitrary hidden sizes or more layers, use the default SimpleCell (without `--use-aoc-cell`).

```bash
# Valid AOCCell configurations:
python examples/train_mnist.py --hidden-size 16 --use-aoc-cell
python examples/train_mnist.py --hidden-size 48 --use-aoc-cell

# Invalid - will raise ValueError:
python examples/train_mnist.py --hidden-size 32 --use-aoc-cell  # unsupported size
python examples/train_mnist.py --hidden-size 48 --num-layers 2 --use-aoc-cell  # only 1 layer supported
```

## MNIST Classification

Train a DEQ model on MNIST digit classification:

```bash
python examples/train_mnist.py --epochs 20 --hidden-size 16

# With AOCCell hardware simulation:
python examples/train_mnist.py --epochs 20 --hidden-size 16 --use-aoc-cell
```

## Fashion MNIST Classification

Train a DEQ model on Fashion MNIST (10 clothing/accessory classes):

```bash
python examples/train_fmnist.py --epochs 20 --hidden-size 16

# With AOCCell hardware simulation:
python examples/train_fmnist.py --epochs 20 --hidden-size 16 --use-aoc-cell
```

Fashion MNIST classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Regression

Train a DEQ model on 1D function regression. Available datasets:
- `sinusoidal2`: sqrt(|x|) * sin(3πx) - a challenging oscillating function
- `gaussian`: Gaussian/bell curve function

```bash
# With AOCCell hardware simulation:
python examples/train_regression.py --dataset sinusoidal2 --epochs 30 --hidden-size 16 --use-aoc-cell

# Generate a plot of results:
python examples/train_regression.py --dataset gaussian --epochs 30 --hidden-size 16 --use-aoc-cell --plot
```

Use `--help` on any script to see all available options.
