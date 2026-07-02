import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            guesses = model(images)
            preds = guesses.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    sensitivity = (conf_matrix.diagonal() / conf_matrix.sum(axis=1)).mean()
    specificity = (conf_matrix.diagonal() / conf_matrix.sum(axis=0)).mean()

    return {
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision,
        "f1":          f1,
        "conf_matrix": conf_matrix
    }