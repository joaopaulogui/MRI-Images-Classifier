import torch
import numpy as np
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight

from src.services.dataset import get_data_loader
from src.services.transforms import get_test_transforms, get_train_transforms
from src.services.models.densenet import train_densenet, train_densenet_kfold
from src.services.models.resnet import train_resnet, train_resnet_kfold
from src.services.models.squeezenet import train_squeezenet, train_squeezenet_kfold
from src.config import TrainingConfig
from src.utils.logger import Logger
from src.services.controllers.train_controller import train_models

def _get_class_weights(loader, num_classes, device):
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe.
    Essencial quando o dataset é desbalanceado.
    """
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    classes = np.arange(num_classes)

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=all_labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train_pipeline(train_data_dir, test_data_dir, epochs, lr, min_accuracy, num_workers, verbose):
    train_ds = datasets.ImageFolder(root=train_data_dir)
    test_ds = datasets.ImageFolder(root=test_data_dir)
    classes = train_ds.classes

    train_transforms = get_train_transforms(224, 224)
    test_transforms = get_test_transforms(224, 224)

    train_loader = get_data_loader(train_ds, train_transforms, batch_size=32, num_workers=num_workers, shuffle=True)
    test_loader = get_data_loader(test_ds, test_transforms, batch_size=32, num_workers=num_workers, shuffle=False)

    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TrainingConfig(
        epochs=epochs,
        lr=lr, 
        min_accuracy=min_accuracy, 
        num_workers=num_workers, 
        classes=classes,
        class_weights=_get_class_weights(train_loader, num_classes, device),
        num_classes=num_classes,
        logger=Logger("generated/logs", "training"),
        verbose=verbose,
    )

    model_registry = {
        "DenseNet":   {"train_fn": train_densenet,   "train_kfold_fn": train_densenet_kfold},
        "ResNet":     {"train_fn": train_resnet,     "train_kfold_fn": train_resnet_kfold},
        "SqueezeNet": {"train_fn": train_squeezenet, "train_kfold_fn": train_squeezenet_kfold},
    }

    train_models(train_ds, train_loader, test_loader, config, model_registry)