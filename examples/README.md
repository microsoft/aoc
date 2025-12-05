# Examples

Training examples for AOC DEQ models on classification and regression tasks.

## Running the Examples

Both scripts support several command-line options to configure training:
- `--epochs` - number of training epochs
- `--hidden-size` - size of hidden layers
- `--num-layers` - number of DEQ layers (default: 1)
- `--use-aoc-cell` - use AOCCell (hardware digital twin) instead of SimpleCell
- `--fixed-point-init` - initialization for fixed-point solver: `zeros`, `x_proj`, or `random`

## MNIST Classification

Train a DEQ model on MNIST digit classification:

```bash
python examples/train_mnist.py --epochs 20 --hidden-size 32
```

## Regression

Train a DEQ model on 1D function regression:

```bash
python examples/train_regression.py --dataset sinusoidal --epochs 50
```

Use `--help` on either script to see all available options.
