import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score, roc_auc_score

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            guesses = model(images)
            probs = F.softmax(guesses, dim=1)
            preds = guesses.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    sensitivity = (conf_matrix.diagonal() / conf_matrix.sum(axis=1)).mean()
    specificity = (conf_matrix.diagonal() / conf_matrix.sum(axis=0)).mean()

    num_classes = len(set(all_labels))
    if num_classes == 2:
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "conf_matrix": conf_matrix,
    }