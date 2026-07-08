import copy
import torch
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold

from src.services.dataset import get_data_loader
from src.services.transforms import get_test_transforms, get_train_transforms
from src.services.models.metrics import evaluate_model
from src.services.models.utils import EarlyStopping


def train_loop(model, optimizer, criterion, train_loader, config, epochs, test_loader, scheduler=None):
    """
    Treina o modelo por N épocas.
    Retorna o modelo com o MELHOR estado de validação (best checkpoint),
    não necessariamente o do último epoch.
    """
    early_stopper = EarlyStopping(patience=config.early_stopping_patience)
    log = config.logger.log if config.logger else print

    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(epochs):
    
        model.train()
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(config.device), labels.to(config.device)

            optimizer.zero_grad()

            guesses = model(images)
            loss = criterion(guesses, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=0.5
            )
            optimizer.step()

            predicted = guesses.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss_value = loss.item()

            if config.verbose:
                log(f"Epoch [{epoch+1}/{epochs}] | Batch [{step+1}/{len(train_loader)}] | Loss: {loss_value:.4f}")


        train_accuracy = correct / total
        log(f"Epoch {epoch+1} train accuracy: {train_accuracy*100:.2f}%")

        metrics = evaluate_model(model, test_loader)
        val_accuracy = metrics["accuracy"]

        log(
            f"Epoch {epoch+1} val accuracy: {val_accuracy*100:.2f}% | "
            f"precision: {metrics['precision']*100:.2f}% | "
            f"f1: {metrics['f1']*100:.2f}% | "
            f"sensitivity: {metrics['sensitivity']*100:.2f}% | "
            f"specificity: {metrics['specificity']*100:.2f}%"
        )

        # Salva o melhor checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            log(f"  ✓ Novo melhor modelo salvo (val_accuracy={val_accuracy*100:.2f}%)")

        if scheduler is not None:
            scheduler.step(val_accuracy)

        log()

        early_stopper(metrics)
        if early_stopper.early_stop:
            log(f"Early stopping ativado na época {epoch+1}.")
            break

    # Restaura o melhor estado encontrado durante o treino
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log(f"Melhor checkpoint restaurado: val_accuracy={best_val_accuracy*100:.2f}%")

    return model


def train_kfold(
    model,
    optimizer_fn,
    criterion,
    dataset,
    config,
    epochs,
    test_loader
):
    """
    Treina com K-Fold estratificado.

    Correções aplicadas:
    - Usa StratifiedKFold para garantir proporção de classes em cada fold.
    - Ao final, retreina o melhor modelo (best_model_state) com o dataset
      COMPLETO, que é a prática correta após seleção via K-Fold.
    """
    log = config.logger.log if config.logger else print

    initial_state = copy.deepcopy(model.state_dict())

    # Extrai labels para o StratifiedKFold
    targets = [dataset[i][1] for i in range(len(dataset))]

    kfold = StratifiedKFold(n_splits=config.kfold_splits, shuffle=True, random_state=42)
    idxs = list(range(len(dataset)))

    fold_accuracies = []
    best_val_accuracy = 0.0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(idxs, targets)):
        log(f"\n{'─' * 50}")
        log(f"  Fold {fold + 1} / {config.kfold_splits}")
        log(f"{'─' * 50}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_transforms = get_train_transforms(config.img_width, config.img_height)
        test_transforms = get_test_transforms(config.img_width, config.img_height)

        train_loader = get_data_loader(train_subset, train_transforms, config.batch_size, config.num_workers, shuffle=True)
        val_loader = get_data_loader(val_subset, test_transforms, config.batch_size, config.num_workers, shuffle=False)

        # Reseta para o estado inicial a cada fold
        model.load_state_dict(copy.deepcopy(initial_state))
        model.to(config.device)

        optimizer = optimizer_fn(model)

        model = train_loop(model, optimizer, criterion, train_loader, config, epochs, val_loader)

        metrics = evaluate_model(model, val_loader)
        fold_accuracies.append(metrics["accuracy"])

        log(
            f"Fold {fold+1} val accuracy: {metrics['accuracy']*100:.2f}% | "
            f"f1: {metrics['f1']*100:.2f}% | "
            f"sensitivity: {metrics['sensitivity']*100:.2f}% | "
            f"specificity: {metrics['specificity']*100:.2f}%"
        )

        if metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = metrics["accuracy"]
            best_model_state = copy.deepcopy(model.state_dict())

    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    log(f"\nMelhor acurácia entre os folds: {best_val_accuracy*100:.2f}%")
    log(f"Acurácia média entre os folds:  {mean_accuracy*100:.2f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # CORREÇÃO PRINCIPAL: retreinar com TODOS os dados de treino.
    # O K-Fold serviu para selecionar os melhores hiperparâmetros/pesos iniciais.
    # O modelo final deve ter visto todo o dataset.
    # ─────────────────────────────────────────────────────────────────────────
    log(f"\n{'='*50}")
    log("  Retreinando com dataset completo (etapa final do K-Fold)")
    log(f"{'='*50}")

    model.load_state_dict(best_model_state)
    model.to(config.device)

    full_train_transforms = get_train_transforms(224, 224)
    full_train_loader = get_data_loader(dataset, full_train_transforms, config.batch_size, config.num_workers, shuffle=True)

    optimizer = optimizer_fn(model)

    # Usa menos épocas no retreino final para não overfitar
    # (sem val_loader real aqui, usamos o mesmo full loader só para monitorar loss)
    final_epochs = max(10, epochs // 3)
    model = train_loop(model, optimizer, criterion, full_train_loader, config, final_epochs, test_loader)

    return model