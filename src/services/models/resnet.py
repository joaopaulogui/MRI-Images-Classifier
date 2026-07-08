import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from src.services.models.trainer import train_loop, train_kfold
from src.services.models.metrics import evaluate_model

def _build_optimizer(model, lr):

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    return optim.Adam(
        [
            {"params": base_model.layer4.parameters(), "lr": lr / 10},
            {"params": base_model.fc.parameters(), "lr": lr},
        ],
        weight_decay=1e-3,
    )

def setup_resnet(device, num_classes):

    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    for param in resnet.parameters():
        param.requires_grad = False
    
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    
    num_in_features = resnet.fc.in_features

    resnet.fc = nn.Sequential(
        nn.Linear(num_in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.6),
        nn.Linear(512, num_classes),
    )

    resnet = resnet.to(device)

    return resnet

def train_resnet(train_loader, test_loader, config, epochs=10, lr=0.001, model=None):

    log = config.logger.log if config.logger else print

    if model is None:
        resnet = setup_resnet(config.device, config.num_classes)
    else:
        resnet = model

    criterion = nn.CrossEntropyLoss(weight=config.class_weights, label_smoothing=0.1)
    
    optimizer = _build_optimizer(resnet, lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.reduce_lr_patience, factor=0.3, min_lr=1e-7
    )

    resnet = train_loop(
        resnet,
        optimizer, 
        criterion, 
        train_loader, 
        config,
        epochs, 
        test_loader, 
        scheduler, 
    )

    metrics = evaluate_model(resnet, test_loader)

    log(
        f"[ResNet] Final → accuracy: {metrics['accuracy']*100:.2f}% | "
        f"f1: {metrics['f1']*100:.2f}% | "
        f"sensitivity: {metrics['sensitivity']*100:.2f}%"
    )

    return resnet, metrics["accuracy"]

def train_resnet_kfold(dataset, test_loader, config, epochs=10, lr=0.001, model=None):

    log = config.logger.log if config.logger else print

    if model is None:
        resnet = setup_resnet(config.device, config.num_classes)
    else:
        resnet = model

    all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
    classes = np.arange(config.num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=all_labels)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    resnet = train_kfold(
        model=resnet,
        optimizer_fn=lambda model: _build_optimizer(model, lr),
        criterion=criterion,
        dataset=dataset,
        config=config,
        epochs=epochs, 
        test_loader=test_loader
    )

    metrics = evaluate_model(resnet, test_loader)

    log(
        f"[resNet K-Fold] Final → accuracy: {metrics['accuracy']*100:.2f}% | "
        f"precision: {metrics['precision']*100:.2f}% | "
        f"f1: {metrics['f1']*100:.2f}% | "
        f"sensitivity: {metrics['sensitivity']*100:.2f}% | "
        f"specificity: {metrics['specificity']*100:.2f}%"
    )

    return resnet, metrics["accuracy"]