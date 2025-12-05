"""
Training and evaluation functions.
Ported from AnalogDEQ for reproducibility.
"""
from typing import Optional, Tuple, List, Dict, Any, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import (
    build_mnist_dataset,
    build_fashion_mnist_dataset,
    get_regression_dataset,
    REGRESSION_DATASETS,
)
from .metrics import (
    Metrics,
    MetricsClassification,
    MetricsRegression,
    ExecutionMode,
)


def get_data_loader(
    dataset_name: str,
    batch_size: int = 32,
    batch_size_test: int = 64,
    train_valid_split_ratio: float = 0.8,
    num_workers: int = 0,
    pin_memory: bool = False,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'fashion_mnist', or regression datasets)
        batch_size: Batch size for training/validation
        batch_size_test: Batch size for testing
        train_valid_split_ratio: Ratio of training data to use for training (rest for validation)
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for GPU transfer
        **dataset_kwargs: Additional arguments passed to dataset builder
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    if dataset_name in ["mnist", "mnist_small"]:
        small = dataset_name == "mnist_small"
        dataset_train, dataset_valid, dataset_test = build_mnist_dataset(
            train_valid_split_ratio=train_valid_split_ratio,
            rescale_to_one_minus_one=dataset_kwargs.get("rescale_to_one_minus_one", False),
            small=small,
        )
    elif dataset_name == "fashion_mnist":
        dataset_train, dataset_valid, dataset_test = build_fashion_mnist_dataset(
            train_valid_split_ratio=train_valid_split_ratio,
        )
    elif dataset_name in REGRESSION_DATASETS:
        n_points = dataset_kwargs.get("n_points", 1000)
        dataset_train = get_regression_dataset(
            dataset_name, n_points=n_points, random_points=False
        )
        dataset_valid = dataset_train  # Use same data for validation
        dataset_test = get_regression_dataset(
            dataset_name, n_points=n_points, random_points=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Choose from: mnist, mnist_small, fashion_mnist, {REGRESSION_DATASETS}")
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size_test,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, valid_loader, test_loader


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "Adam",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer (Adam, AdamW, SGD, etc.)
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_name not in torch.optim.__dict__:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    optimizer_class = torch.optim.__dict__[optimizer_name]
    return optimizer_class(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **kwargs,
    )


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: Optional[str] = None,
    **kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        scheduler_name: Name of the scheduler (CosineAnnealingLR, StepLR, etc.)
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_name is None:
        return None
    
    if scheduler_name not in torch.optim.lr_scheduler.__dict__:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    scheduler_class = torch.optim.lr_scheduler.__dict__[scheduler_name]
    return scheduler_class(optimizer, **kwargs)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: str = "classification",
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    clip_grad_norm: Optional[float] = None,
    fixed_point_init: str = "x_proj",
    show_progress: bool = True,
) -> Union[MetricsClassification, MetricsRegression]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        task_type: 'classification' or 'regression'
        lr_scheduler: Optional learning rate scheduler (step per batch)
        clip_grad_norm: Optional gradient clipping norm
        fixed_point_init: Fixed point initialization method
        show_progress: Whether to show progress bar
        
    Returns:
        Metrics object with training statistics
    """
    model.train()
    
    if task_type == "classification":
        metrics = MetricsClassification(mode=ExecutionMode.TRAIN, n_classes=10)
    else:
        metrics = MetricsRegression(mode=ExecutionMode.TRAIN)
    
    loader = tqdm(train_loader, desc="Training") if show_progress else train_loader
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, iter_info = model(data, fixed_point_init=fixed_point_init)
        
        # Compute loss
        if task_type == "classification":
            loss = loss_fn(output, target)
        else:
            loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # Update learning rate scheduler (if per-batch)
        current_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Update metrics
        if task_type == "classification":
            metrics.update(
                loss=loss.item(),
                output=output,
                target=target,
                iter_info=iter_info,
                lr=current_lr,
            )
        else:
            metrics.update(
                loss=loss.item(),
                output=output,
                target=target,
                iter_info=iter_info,
                lr=current_lr,
            )
        
        if show_progress:
            loader.set_postfix(loss=loss.item())
    
    metrics.finalize()
    return metrics


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    task_type: str = "classification",
    mode: ExecutionMode = ExecutionMode.VALID,
    fixed_point_init: str = "x_proj",
    show_progress: bool = True,
) -> Union[MetricsClassification, MetricsRegression]:
    """
    Evaluate the model for one epoch.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader
        loss_fn: Loss function
        device: Device to evaluate on
        task_type: 'classification' or 'regression'
        mode: Evaluation mode (VALID or TEST)
        fixed_point_init: Fixed point initialization method
        show_progress: Whether to show progress bar
        
    Returns:
        Metrics object with evaluation statistics
    """
    model.eval()
    
    if task_type == "classification":
        metrics = MetricsClassification(mode=mode, n_classes=10)
    else:
        metrics = MetricsRegression(mode=mode)
    
    desc = "Validation" if mode == ExecutionMode.VALID else "Testing"
    loader = tqdm(data_loader, desc=desc) if show_progress else data_loader
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        output, iter_info = model(data, fixed_point_init=fixed_point_init)
        
        if task_type == "classification":
            loss = loss_fn(output, target)
        else:
            loss = loss_fn(output, target)
        
        if task_type == "classification":
            metrics.update(
                loss=loss.item(),
                output=output,
                target=target,
                iter_info=iter_info,
            )
        else:
            metrics.update(
                loss=loss.item(),
                output=output,
                target=target,
                iter_info=iter_info,
            )
        
        if show_progress:
            loader.set_postfix(loss=loss.item())
    
    metrics.finalize()
    return metrics
