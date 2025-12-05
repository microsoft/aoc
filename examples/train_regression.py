"""
Example: Train an AOC DEQ model on 1D function regression.

This script demonstrates how to train a Deep Equilibrium (DEQ) model
using the AOC digital twin on a 1D function regression task.

Available regression functions:
- sinusoidal: x * cos(x) scaled to [-1, 1]
- sinusoidal2: sqrt(|x|) * sin(3*pi*x)
- polynomial: 5th order polynomial
- gaussian: Gaussian function

Usage:
    python examples/train_regression.py --dataset sinusoidal --epochs 50

The training pipeline is ported from AnalogDEQ for reproducibility.
"""
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from aoc import DEQInputOutputProjection
from aoc.training import (
    get_data_loader,
    get_optimizer,
    get_lr_scheduler,
    train_epoch,
    eval_epoch,
    ExecutionMode,
    REGRESSION_DATASETS,
    get_regression_function,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AOC model on regression")
    parser.add_argument("--dataset", type=str, default="sinusoidal",
                       choices=REGRESSION_DATASETS, help="Regression dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of layers (d_hidden will have num_layers+1 elements)")
    parser.add_argument("--n-points", type=int, default=1000, help="Number of data points")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-aoc-cell", action="store_true", help="Use AOCCell instead of SimpleCell")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--fixed-point-init", type=str, default="zeros",
                       choices=["x_proj", "zeros", "random"],
                       help="Fixed point initialization: x_proj, zeros, or random")
    parser.add_argument("--plot", action="store_true", help="Plot results after training")
    return parser.parse_args()


def plot_regression_results(model, dataset_name, device, fixed_point_init="zeros", save_path="regression_result.png"):
    """Plot the learned function vs ground truth."""
    model.eval()
    
    # Generate dense x values for plotting
    x = torch.linspace(-1, 1, 200).unsqueeze(1)
    
    # Get ground truth
    func = get_regression_function(dataset_name)
    y_true = func(x.squeeze())
    
    # Get model predictions
    with torch.no_grad():
        x_device = x.to(device)
        y_pred, _ = model(x_device, fixed_point_init=fixed_point_init)
        y_pred = y_pred.cpu().squeeze()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y_true.numpy(), 'b-', label='Ground Truth', linewidth=2)
    plt.plot(x.numpy(), y_pred.numpy(), 'r--', label='DEQ Prediction', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Regression on {dataset_name} function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print(f"Loading {args.dataset} regression dataset with {args.n_points} points...")
    train_loader, valid_loader, test_loader = get_data_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        batch_size_test=args.n_points,  # Evaluate on all points
        n_points=args.n_points,
    )
    print(f"Train batches: {len(train_loader)}")
    
    # Create model
    # Regression: 1 input, 1 output
    d_in = 1
    # d_hidden has num_layers+1 elements: [h, h] for 1 layer, [h, h, h] for 2 layers, etc.
    d_hidden = [args.hidden_size] * (args.num_layers + 1)
    d_out = 1
    
    if args.use_aoc_cell:
        print(f"Creating AOCCell model with {args.num_layers} layer(s), hidden size {args.hidden_size}...")
        model = DEQInputOutputProjection.create_default_aoc_model(
            d_in=d_in,
            d_hidden=d_hidden,
            d_out=d_out,
            # connectivity defaults to FEEDBACK (DEQ)
        )
    else:
        print(f"Creating SimpleCell model with {args.num_layers} layer(s), hidden size {args.hidden_size}...")
        model = DEQInputOutputProjection.create_simple_cell_model(
            d_in=d_in,
            d_hidden=d_hidden,
            d_out=d_out,
        )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    
    # Print network architecture
    print(f"\nNetwork Architecture:")
    print(f"  Input:  {d_in}")
    print(f"  Hidden: {d_hidden}")
    print(f"  Output: {d_out}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Fixed-point init: {args.fixed_point_init}")
    print(f"\nModel:\n{model}")
    
    # Loss function, optimizer
    loss_fn = nn.MSELoss()
    optimizer = get_optimizer(model, optimizer_name="Adam", lr=args.lr, weight_decay=0)
    
    # Optional: learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_name="CosineAnnealingLR",
        T_max=args.epochs * len(train_loader),
    )
    
    # Training loop
    best_valid_loss = float('inf')
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            task_type="regression",
            lr_scheduler=scheduler,
            clip_grad_norm=args.clip_grad,
            fixed_point_init=args.fixed_point_init,
            show_progress=False,
        )
        
        # Validate
        valid_metrics = eval_epoch(
            model=model,
            data_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            task_type="regression",
            mode=ExecutionMode.VALID,
            fixed_point_init=args.fixed_point_init,
            show_progress=False,
        )
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{args.epochs}: "
                  f"Train MSE={train_metrics.mse_avg:.6f}, "
                  f"Valid MSE={valid_metrics.mse_avg:.6f}, "
                  f"Corr={valid_metrics.pearson_corr:.4f}")
        
        # Track best model
        if valid_metrics.mse_avg < best_valid_loss:
            best_valid_loss = valid_metrics.mse_avg
    
    print("\n" + "=" * 60)
    print("Training complete!")
    
    # Final test evaluation
    print("\nFinal test evaluation...")
    test_metrics = eval_epoch(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        task_type="regression",
        mode=ExecutionMode.TEST,
        fixed_point_init=args.fixed_point_init,
        show_progress=False,
    )
    print(f"Test: {test_metrics}")
    print(f"\nFinal test MSE: {test_metrics.mse_avg:.6f}")
    print(f"Final Pearson correlation: {test_metrics.pearson_corr:.4f}")
    
    # Plot results
    if args.plot:
        plot_regression_results(model, args.dataset, device, args.fixed_point_init)


if __name__ == "__main__":
    main()
