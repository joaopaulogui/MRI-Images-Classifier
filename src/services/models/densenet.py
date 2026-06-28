import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.services.models.trainer import train_loop, train_kfold
from src.services.models.metrics import evaluate_model


def setup_densenet(device, num_classes):
    """
    Carrega DenseNet201 pré-treinado e prepara para fine-tuning.

    Correções aplicadas:
    - Adiciona BatchNorm + Dropout na cabeça classificadora para regularização.
    """
    densenet = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)

    # Congela tudo primeiro
    for param in densenet.parameters():
        param.requires_grad = False

    # Descongela denseblock4 (mais capacidade de adaptação)

    for param in densenet.features.denseblock4.parameters():
        param.requires_grad = True

    # Cabeça classificadora com Dropout para regularização
    num_in_features = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(
        nn.Linear(num_in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )

    densenet = densenet.to(device)

    return densenet


def _build_optimizer(model, lr):
    """
    Otimizador com grupos de LR diferenciados por profundidade.
    - denseblock4: LR baixo
    - classifier: LR normal
    """

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    return optim.Adam(
        [
            {"params": base_model.features.denseblock4.parameters(), "lr": lr / 10},
            {"params": base_model.classifier.parameters(),           "lr": lr},
        ],
        weight_decay=1e-3,
    )


def train_densenet(train_loader, test_loader, config, epochs=50, lr=0.001, model=None):

    log = config.logger.log if config.logger else print

    if model is None:
        densenet = setup_densenet(config.device, config.num_classes)
    else:
        densenet = model

    # CrossEntropy com pesos por classe
    criterion = nn.CrossEntropyLoss(weight=config.class_weights, label_smoothing=0.1)

    optimizer = _build_optimizer(densenet, lr)

    # ReduceLROnPlateau com patience maior para não decair LR rápido demais
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.reduce_lr_patience, factor=0.3, min_lr=1e-7
    )

    densenet = train_loop(
        densenet, 
        optimizer, 
        criterion, 
        train_loader, 
        config, 
        epochs, 
        test_loader, 
        scheduler,
    )

    metrics = evaluate_model(densenet, test_loader)

    log(
        f"[DenseNet] Final → accuracy: {metrics['accuracy']*100:.2f}% | "
        f"f1: {metrics['f1']*100:.2f}% | "
        f"sensitivity: {metrics['sensitivity']*100:.2f}%"
    )

    return densenet, metrics["accuracy"]


def train_densenet_kfold(dataset, test_loader, config, epochs=10, lr=0.001, model=None):
 
    log = config.logger.log if config.logger else print

    if model is None:
        densenet = setup_densenet(config.device, config.num_classes)
    else:
        densenet = model

    # Para K-Fold calculamos os pesos sobre o dataset completo
    all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
    classes = np.arange(config.num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=all_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    densenet = train_kfold(
        model=densenet,
        optimizer_fn=lambda m: _build_optimizer(m, lr),
        criterion=criterion,
        dataset=dataset,
        config=config,
        epochs=epochs,
        test_loader=test_loader
    )

    metrics = evaluate_model(densenet, test_loader)

    log(
        f"[DenseNet K-Fold] Final → accuracy: {metrics['accuracy']*100:.2f}% | "
        f"precision: {metrics['precision']*100:.2f}% | "
        f"f1: {metrics['f1']*100:.2f}% | "
        f"sensitivity: {metrics['sensitivity']*100:.2f}% | "
        f"specificity: {metrics['specificity']*100:.2f}%"
    )

    return densenet, metrics["accuracy"]