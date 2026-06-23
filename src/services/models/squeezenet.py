import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.services.models.trainer import train_loop, train_kfold
from src.services.models.metrics import evaluate_model

def _build_optimizer(model, lr):
    """
    Otimizador com grupos de LR diferenciados por profundidade.
    - transition3: LR muito baixo (features já bem treinadas)
    - denseblock4: LR baixo
    - classifier: LR normal
    """

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    return optim.Adam(
        [
            {"params": base_model.features[10].transition3.parameters(), "lr": lr / 100},
            {"params": base_model.features[12].denseblock4.parameters(), "lr": lr / 10},
            {"params": base_model.classifier[1].parameters(), "lr": lr},
        ],
        weight_decay=1e-4,
    )

def setup_squeezenet(device, num_classes):

    squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)

    #Freeze parameters for Transfer Learning
    for param in squeezenet.parameters():
        param.requires_grad = False

    #Unreeze last features for better learning
    for param in squeezenet.features[-3:].parameters():
        param.requires_grad = True

    num_in_channels = squeezenet.classifier[1].in_channels

    squeezenet.classifier[1] = nn.Conv2d(num_in_channels, num_classes, kernel_size=(1, 1))

    if torch.cuda.device_count() > 1:
        squeezenet = nn.DataParallel(squeezenet)

    squeezenet = squeezenet.to(device)

    return squeezenet

def train_squeezenet(train_loader, test_loader, num_classes, epochs=10, lr=0.001, model=None, verbose=True, logger=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        squeezenet = setup_squeezenet(device, num_classes)
    else:
        squeezenet = model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": squeezenet.features[10].parameters(), "lr": lr/100},
        {"params": squeezenet.features[12].parameters(), "lr": lr/100},
        {"params": squeezenet.classifier[1].parameters(), "lr": lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.1)

    squeezenet = train_loop(squeezenet, optimizer, criterion, train_loader, epochs, device, test_loader, scheduler, verbose=verbose, logger=logger)

    metrics = evaluate_model(squeezenet, test_loader)

    return squeezenet, metrics["accuracy"]

def train_squeezenet_kfold(dataset, test_loader, num_classes, epochs=10, lr=0.001, model=None, num_workers=2, verbose=True, logger=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        squeezenet = setup_squeezenet(device, num_classes)
    else:
        squeezenet = model

    criterion = nn.CrossEntropyLoss()

    squeezenet = train_kfold(
        model=squeezenet,
        optimizer_fn=lambda model: _build_optimizer(model, lr),
        criterion=criterion,
        dataset=dataset,
        epochs=epochs, 
        device=device,
        num_workers=num_workers,
        verbose=verbose,
        logger=logger
    )

    metrics = evaluate_model(squeezenet, test_loader)

    return squeezenet, metrics["accuracy"]