"""
Example: Train an AOC DEQ model on Fashion MNIST classification.

This script demonstrates how to train a Deep Equilibrium (DEQ) model
using the AOC digital twin on the Fashion MNIST clothing classification task.

Fashion MNIST classes: T-shirt/top, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Sneaker, Bag, Ankle boot.

Usage:
    python examples/train_fmnist.py --epochs 20 --lr 3e-4

The training pipeline is ported from AnalogDEQ for reproducibility.
"""
import argparse
import torch
import torch.nn as nn

from aoc import DEQInputOutputProjection
from aoc.training import (
    get_data_loader,
    get_optimizer,
    get_lr_scheduler,
    train_epoch,
    eval_epoch,
    ExecutionMode,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AOC model on Fashion MNIST")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of layers (d_hidden will have num_layers+1 elements)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-aoc-cell", action="store_true", help="Use AOCCell instead of SimpleCell")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--fixed-point-init", type=str, default="zeros",
                       choices=["x_proj", "zeros", "random"],
                       help="Fixed point initialization: x_proj, zeros, or random")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading Fashion MNIST dataset...")
    train_loader, valid_loader, test_loader = get_data_loader(
        dataset_name="fashion_mnist",
        batch_size=args.batch_size,
        batch_size_test=256,
        train_valid_split_ratio=0.9,
    )
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    
    # Create model
    # Fashion MNIST: 28x28 = 784 input features, 10 output classes
    d_in = 784
    # d_hidden has num_layers+1 elements: [h, h] for 1 layer, [h, h, h] for 2 layers, etc.
    d_hidden = [args.hidden_size] * (args.num_layers + 1)
    d_out = 10
    
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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name="Adam", lr=args.lr, weight_decay=0)
    
    # Optional: learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_name="CosineAnnealingLR",
        T_max=args.epochs * len(train_loader),
    )
    
    # Training loop
    best_valid_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            task_type="classification",
            lr_scheduler=scheduler,
            clip_grad_norm=args.clip_grad,
            fixed_point_init=args.fixed_point_init,
            show_progress=True,
        )
        print(f"  Train: {train_metrics}")
        
        # Validate
        valid_metrics = eval_epoch(
            model=model,
            data_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            task_type="classification",
            mode=ExecutionMode.VALID,
            fixed_point_init=args.fixed_point_init,
            show_progress=False,
        )
        print(f"  Valid: {valid_metrics}")
        
        # Track best model
        if valid_metrics.accuracy > best_valid_acc:
            best_valid_acc = valid_metrics.accuracy
            print(f"  [New best validation accuracy: {best_valid_acc:.4f}]")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    
    # Final test evaluation
    print("\nFinal test evaluation...")
    test_metrics = eval_epoch(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        task_type="classification",
        mode=ExecutionMode.TEST,
        fixed_point_init=args.fixed_point_init,
        show_progress=True,
    )
    print(f"Test: {test_metrics}")
    print(f"\nFinal test accuracy: {test_metrics.accuracy:.4f}")
    
    # Print class info
    print("\nFashion MNIST classes:")
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")


if __name__ == "__main__":
    main()
