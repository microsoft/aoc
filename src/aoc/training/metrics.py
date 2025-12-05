"""
Metrics for training and evaluation.
Ported from AnalogDEQ for reproducibility.
"""
from typing import Optional, Dict, List, Any
from enum import Enum
import torch
import numpy as np


class ExecutionMode(Enum):
    """Execution mode for training/validation/test."""
    TRAIN = 0
    VALID = 1
    TEST = 2


def calculate_avg_std(data: List[float]) -> tuple:
    """Calculate average and standard deviation of a list."""
    if len(data) == 0:
        return 0.0, 0.0
    arr = np.array(data)
    return float(np.mean(arr)), float(np.std(arr))


class Metrics:
    """
    Base metrics class for tracking training progress.
    """

    def __init__(self, mode: ExecutionMode, problem: str = ""):
        self.mode = mode
        self.problem = problem
        self.loss_log: List[float] = []
        self.reg_loss_log: List[float] = []
        self.lr_log: List[float] = []
        
        # Solver iteration metrics
        self.f_nstep_log: List[float] = []
        self.f_lowest_rel_log: List[float] = []
        
        # Final aggregated values
        self.loss_avg, self.loss_std = 0.0, 0.0
        self.reg_loss_avg, self.reg_loss_std = 0.0, 0.0
        self.f_nstep_avg, self.f_nstep_std = 0.0, 0.0
        self.f_lowest_rel_avg, self.f_lowest_rel_std = 0.0, 0.0
        self.avg_lr = 0.0
        
        # Metric used for checkpointing (lower is better)
        self.checkpoint_save_metric = float('inf')

    def update(
        self,
        loss: float,
        reg_loss: Optional[float] = None,
        iter_info: Optional[Dict[str, Any]] = None,
        lr: float = 0.0,
    ):
        """
        Update metrics with batch results.
        
        Args:
            loss: The main loss value
            reg_loss: Optional regularization loss
            iter_info: Optional dictionary with DEQ iteration info from torchdeq
            lr: Current learning rate
        """
        self.loss_log.append(loss)
        self.reg_loss_log.append(reg_loss if reg_loss is not None else 0.0)
        self.lr_log.append(lr)
        
        # Extract iteration info if available (from torchdeq)
        if iter_info is not None:
            nstep = iter_info.get("nstep", 0)
            if isinstance(nstep, torch.Tensor):
                nstep = nstep.mean().item()
            self.f_nstep_log.append(float(nstep))
            
            lowest = iter_info.get("rel_lowest", iter_info.get("abs_lowest", 0))
            if isinstance(lowest, torch.Tensor):
                lowest = lowest.mean().item()
            self.f_lowest_rel_log.append(float(lowest))

    def finalize(self):
        """Compute final aggregated metrics."""
        self.loss_avg, self.loss_std = calculate_avg_std(self.loss_log)
        self.reg_loss_avg, self.reg_loss_std = calculate_avg_std(self.reg_loss_log)
        self.f_nstep_avg, self.f_nstep_std = calculate_avg_std(self.f_nstep_log)
        self.f_lowest_rel_avg, self.f_lowest_rel_std = calculate_avg_std(self.f_lowest_rel_log)
        self.avg_lr = np.mean(self.lr_log) if self.lr_log else 0.0
        self.checkpoint_save_metric = self.loss_avg

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for logging."""
        prefix = self.mode.name.lower()
        return {
            f"{prefix}/loss": self.loss_avg,
            f"{prefix}/loss_std": self.loss_std,
            f"{prefix}/reg_loss": self.reg_loss_avg,
            f"{prefix}/nstep": self.f_nstep_avg,
            f"{prefix}/convergence": self.f_lowest_rel_avg,
            f"{prefix}/lr": self.avg_lr,
        }

    def __str__(self) -> str:
        return (
            f"{self.mode.name}: loss={self.loss_avg:.4f}±{self.loss_std:.4f}, "
            f"nstep={self.f_nstep_avg:.1f}, convergence={self.f_lowest_rel_avg:.2e}"
        )


class MetricsClassification(Metrics):
    """
    Metrics for classification tasks.
    Tracks accuracy in addition to base metrics.
    """

    def __init__(self, mode: ExecutionMode, n_classes: int = 10, problem: str = ""):
        super().__init__(mode, problem)
        self.n_classes = n_classes
        self.correct_log: List[int] = []
        self.total_log: List[int] = []
        self.accuracy = 0.0

    def update(
        self,
        loss: float,
        output: torch.Tensor,
        target: torch.Tensor,
        reg_loss: Optional[float] = None,
        iter_info: Optional[Dict[str, Any]] = None,
        lr: float = 0.0,
    ):
        """
        Update classification metrics.
        
        Args:
            loss: The main loss value
            output: Model output logits (batch_size, n_classes)
            target: Ground truth labels (batch_size,)
            reg_loss: Optional regularization loss
            iter_info: Optional DEQ iteration info
            lr: Current learning rate
        """
        super().update(loss, reg_loss, iter_info, lr)
        
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        total = target.size(0)
        
        self.correct_log.append(correct)
        self.total_log.append(total)

    def finalize(self):
        """Compute final classification metrics including accuracy."""
        super().finalize()
        total_correct = sum(self.correct_log)
        total_samples = sum(self.total_log)
        self.accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        # For classification, use negative accuracy as checkpoint metric (higher accuracy = lower metric)
        self.checkpoint_save_metric = 1.0 - self.accuracy

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        d = super().to_dict()
        prefix = self.mode.name.lower()
        d[f"{prefix}/accuracy"] = self.accuracy
        return d

    def __str__(self) -> str:
        return (
            f"{self.mode.name}: loss={self.loss_avg:.4f}, acc={self.accuracy:.4f}, "
            f"nstep={self.f_nstep_avg:.1f}"
        )


class MetricsRegression(Metrics):
    """
    Metrics for regression tasks.
    Tracks MSE and Pearson correlation.
    """

    def __init__(self, mode: ExecutionMode, problem: str = ""):
        super().__init__(mode, problem)
        self.mse_log: List[float] = []
        self.predictions: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.mse_avg = 0.0
        self.pearson_corr = 0.0

    def update(
        self,
        loss: float,
        output: torch.Tensor,
        target: torch.Tensor,
        reg_loss: Optional[float] = None,
        iter_info: Optional[Dict[str, Any]] = None,
        lr: float = 0.0,
    ):
        """
        Update regression metrics.
        
        Args:
            loss: The main loss value (typically MSE)
            output: Model predictions
            target: Ground truth values
            reg_loss: Optional regularization loss
            iter_info: Optional DEQ iteration info
            lr: Current learning rate
        """
        super().update(loss, reg_loss, iter_info, lr)
        
        mse = torch.nn.functional.mse_loss(output, target).item()
        self.mse_log.append(mse)
        self.predictions.append(output.detach().cpu())
        self.targets.append(target.detach().cpu())

    def finalize(self):
        """Compute final regression metrics."""
        super().finalize()
        self.mse_avg, _ = calculate_avg_std(self.mse_log)
        
        # Compute Pearson correlation
        if self.predictions and self.targets:
            all_preds = torch.cat(self.predictions).flatten().numpy()
            all_targets = torch.cat(self.targets).flatten().numpy()
            if len(all_preds) > 1:
                self.pearson_corr = float(np.corrcoef(all_preds, all_targets)[0, 1])
            else:
                self.pearson_corr = 0.0
        
        self.checkpoint_save_metric = self.mse_avg

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        d = super().to_dict()
        prefix = self.mode.name.lower()
        d[f"{prefix}/mse"] = self.mse_avg
        d[f"{prefix}/pearson_corr"] = self.pearson_corr
        return d

    def __str__(self) -> str:
        return (
            f"{self.mode.name}: loss={self.loss_avg:.4f}, mse={self.mse_avg:.4f}, "
            f"corr={self.pearson_corr:.4f}, nstep={self.f_nstep_avg:.1f}"
        )
