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
            {"params": base_model.layer3.parameters(), "lr": lr / 100},
            {"params": base_model.layer4.parameters(), "lr": lr / 10},
            {"params": base_model.fc.parameters(), "lr": lr},
        ],
        weight_decay=1e-4,
    )

def setup_resnet(device, num_classes):

    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    #Freeze parameters for Transfer Learning
    for param in resnet.parameters():
        param.requires_grad = False
    
    #Unfreeze last features for better learning
    for param in resnet.layer4.parameters():
        param.requires_grad = True

    for param in resnet.layer3.parameters():
        param.requires_grad = True
    
    num_in_features = resnet.fc.in_features

    resnet.fc = nn.Linear(num_in_features, num_classes)

    if torch.cuda.device_count() > 1:
        resnet = nn.DataParallel(resnet)

    resnet = resnet.to(device)

    return resnet

def train_resnet(train_loader, test_loader, num_classes, epochs=10, lr=0.001, model = None, verbose=True, logger = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        resnet = setup_resnet(device, num_classes)
    else:
        resnet = model

    criterion = nn.CrossEntropyLoss()

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    optimizer = optim.Adam([
            {"params": base_model.layer3.parameters(), "lr": lr/100},
            {"params": base_model.layer4.parameters(), "lr": lr/100},
            {"params": base_model.fc.parameters(), "lr": lr},
        ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.1)

    resnet = train_loop(resnet, optimizer, criterion, train_loader, epochs, device, test_loader, scheduler, verbose=verbose, logger=logger)

    metrics = evaluate_model(resnet, test_loader)

    return resnet, metrics["accuracy"]

def train_resnet_kfold(dataset, test_loader, num_classes, epochs=10, lr=0.001, model=None, num_workers=2, verbose=True, logger=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        resnet = setup_resnet(device, num_classes)
    else:
        resnet = model

    criterion = nn.CrossEntropyLoss()

    resnet = train_kfold(
        model=resnet,
        optimizer_fn=lambda model: _build_optimizer(model, lr),
        criterion=criterion,
        dataset=dataset,
        epochs=epochs, 
        device=device,
        num_workers=num_workers,
        verbose=verbose,
        logger=logger
    )

    metrics = evaluate_model(resnet, test_loader)

    return resnet, metrics["accuracy"]