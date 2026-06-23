import torch
from dataclasses import dataclass, field

from src.utils.logger import Logger

@dataclass
class TrainingConfig:
    #Training
    epochs: int
    lr: float
    min_accuracy: float
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    
    #Data
    batch_size: int = 32
    num_workers: int = 2
    img_width: int = 224
    img_height: int = 224
    val_split: float = 0.2

    #Labels
    class_weights: torch.Tensor = None
    classes: list[str] = None
    num_classes: int = None

    #KFold
    kfold_splits: int = 5
    
    #Early Stopping
    early_stopping_patience: int = 10
    
    #Environment
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    #Logs
    logger: Logger = field(default_factory=Logger)
    verbose: bool = True