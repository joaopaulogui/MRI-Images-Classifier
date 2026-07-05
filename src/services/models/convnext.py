import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.services.models.trainer import train_loop, train_kfold
from src.services.models.metrics import evaluate_model

def setup_convnext(device, num_classes, variant="tiny"):
    """
    Carrega ConvNeXt pré-treinada (ImageNet) e prepara para fine-tuning.
    variant: "tiny", "small", "base", "large".
    """
    builder = getattr(models, f"convnext_{variant}")
    weights_enum = getattr(models, f"ConvNeXt_{variant.capitalize()}_Weights")
    convnext = builder(weights=weights_enum.DEFAULT)

    # Congela tudo primeiro
    for param in convnext.parameters():
        param.requires_grad = False

    # Descongela o último estágio (mais capacidade de adaptação)
    for param in convnext.features[-1].parameters():
        param.requires_grad = True

    # Cabeça classificadora com Dropout para regularização
    # (classifier original: LayerNorm2d -> Flatten -> Linear)
    num_in_features = convnext.classifier[2].in_features
    convnext.classifier = nn.Sequential(
        convnext.classifier[0],   # mantém o LayerNorm2d original
        convnext.classifier[1],   # mantém o Flatten original
        nn.Linear(num_in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )

    return convnext.to(device)


def _build_optimizer(model, lr):
    """
    Otimizador com grupos de LR diferenciados por profundidade.
    - último estágio (features[-1]): LR baixo
    - classifier: LR normal
    """
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    return optim.Adam(
        [
            {"params": base_model.features[-1].parameters(), "lr": lr / 10},
            {"params": base_model.classifier.parameters(),   "lr": lr},
        ],
        weight_decay=1e-3,
    )


def train_convnext(train_loader, test_loader, config, epochs=50, lr=0.001, model=None, variant="tiny"):
    log = config.logger.log if config.logger else print

    if model is None:
        convnext = setup_convnext(config.device, config.num_classes, variant=variant)
    else:
        convnext = model

    criterion = nn.CrossEntropyLoss(weight=config.class_weights, label_smoothing=0.1)
    optimizer = _build_optimizer(convnext, lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.reduce_lr_patience, factor=0.3, min_lr=1e-7
    )

    convnext = train_loop(
        convnext, optimizer, criterion, train_loader, config, epochs, test_loader, scheduler,
    )

    metrics = evaluate_model(convnext, test_loader)
    log(
        f"[ConvNeXt-{variant}] Final → accuracy: {metrics['accuracy']*100:.2f}% | "
        f"f1: {metrics['f1']*100:.2f}% | "
        f"sensitivity: {metrics['sensitivity']*100:.2f}%"
    )

    return convnext, metrics["accuracy"]

def train_convnext_kfold(dataset, test_loader, config, epochs=50, lr=0.001, model=None, variant="tiny"):
    log = config.logger.log if config.logger else print

    if model is None:
        convnext = setup_convnext(config.device, config.num_classes, variant=variant)
    else:
        convnext = model

    all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
    classes = np.arange(config.num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=all_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    convnext = train_kfold(
        model=convnext,
        optimizer_fn=lambda m: _build_optimizer(m, lr),
        criterion=criterion,
        dataset=dataset,
        config=config,
        epochs=epochs,
        test_loader=test_loader
    )

    metrics = evaluate_model(convnext, test_loader)

    log(
        f"[ConvNeXt K-Fold] Final → accuracy: {metrics['accuracy']*100:.2f}% | "
        f"precision: {metrics['precision']*100:.2f}% | "
        f"f1: {metrics['f1']*100:.2f}% | "
        f"sensitivity: {metrics['sensitivity']*100:.2f}% | "
        f"specificity: {metrics['specificity']*100:.2f}%"
    )

    return convnext, metrics["accuracy"]
