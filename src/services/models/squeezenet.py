import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.services.models.trainer import train_loop, train_kfold
from src.services.models.metrics import evaluate_model

def _build_optimizer(model, lr):
    """
    Otimizador com grupos de LR diferenciados por profundidade.
    - classifier: LR normal
    """

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    return optim.Adam(
        [
            {"params": base_model.features[-6:].parameters(), "lr": lr / 10},
            {"params": base_model.classifier.parameters(), "lr": lr/3},
        ],
        weight_decay=1e-3,
    )

def setup_squeezenet(device, num_classes):

    squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)

    #Freeze parameters for Transfer Learning
    for param in squeezenet.parameters():
        param.requires_grad = False

    #Unreeze last features for better learning
    for param in squeezenet.features[-3:].parameters(): #if it stops learning decrease to [-3:]
        param.requires_grad = True

    num_in_channels = squeezenet.classifier[1].in_channels

    squeezenet.classifier[1] = nn.Sequential(
        nn.Conv2d(num_in_channels, 512, kernel_size=(1, 1)),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, num_classes, kernel_size=(1, 1)),
    )

    squeezenet = squeezenet.to(device)

    return squeezenet

def train_squeezenet(train_loader, test_loader, config, epochs=10, lr=0.001, model=None):

    log = config.logger.log if config.logger else print

    if model is None:
        squeezenet = setup_squeezenet(config.device, config.num_classes)
    else:
        squeezenet = model

    criterion = nn.CrossEntropyLoss(weight=config.class_weights, label_smoothing=0.1)
    optimizer = _build_optimizer(squeezenet, lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.reduce_lr_patience, factor=0.3, min_lr=1e-7
    )

    squeezenet = train_loop(squeezenet, optimizer, criterion, train_loader, config, epochs, test_loader, scheduler)

    metrics = evaluate_model(squeezenet, test_loader)

    log(
        f"[SqueezeNet] Final → accuracy: {metrics['accuracy']*100:.2f}% | "
        f"f1: {metrics['f1']*100:.2f}% | "
        f"sensitivity: {metrics['sensitivity']*100:.2f}%"
    )

    return squeezenet, metrics["accuracy"]

def train_squeezenet_kfold(dataset, test_loader, config, epochs=10, lr=0.001, model=None):

    log = config.logger.log if config.logger else print

    if model is None:
        squeezenet = setup_squeezenet(config.device, config.num_classes)
    else:
        squeezenet = model

    criterion = nn.CrossEntropyLoss(weight=config.class_weights, label_smoothing=0.1)

    squeezenet = train_kfold(
        model=squeezenet,
        optimizer_fn=lambda model: _build_optimizer(model, lr),
        criterion=criterion,
        dataset=dataset,
        config=config,
        epochs=epochs, 
        test_loader=test_loader
    )

    metrics = evaluate_model(squeezenet, test_loader)

    log(
        f"[SqueezeNet K-Fold] Final → accuracy: {metrics['accuracy']*100:.2f}% | "
        f"precision: {metrics['precision']*100:.2f}% | "
        f"f1: {metrics['f1']*100:.2f}% | "
        f"sensitivity: {metrics['sensitivity']*100:.2f}% | "
        f"specificity: {metrics['specificity']*100:.2f}%"
    )

    return squeezenet, metrics["accuracy"]