import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
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
    
    total = conf_matrix.sum()
    specificities = []
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = conf_matrix[i, :].sum() - TP
        FP = conf_matrix[:, i].sum() - TP
        TN = total - TP - FN - FP
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificities.append(spec)

    specificity = np.mean(specificities)

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

def ensemble_predict_argmax(models, best_model_idx, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_final_preds = []
    all_labels = []
    all_avg_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            batch_preds_per_model = []
            batch_probs_per_model = []
            for model in models:
                guesses = model(images)
                probs = F.softmax(guesses, dim=1)
                preds = guesses.argmax(dim=1)

                batch_preds_per_model.append(preds.cpu())
                batch_probs_per_model.append(probs.cpu())

            batch_preds_per_model = torch.stack(batch_preds_per_model, dim=1)
            batch_probs_per_model = torch.stack(batch_probs_per_model, dim=1)

            avg_probs = batch_probs_per_model.mean(dim=1)
            all_avg_probs.append(avg_probs)

            for sample_preds in batch_preds_per_model:
                votes = Counter(sample_preds.tolist())
                most_common = votes.most_common()

                if most_common[0][1] > 1:
                    final_class = most_common[0][0]
                else:
                    final_class = sample_preds[best_model_idx].item()

                all_final_preds.append(final_class)

            all_labels.extend(labels.tolist())
    
    all_avg_probs = torch.cat(all_avg_probs, dim=0).numpy()
    
    accuracy = accuracy_score(all_labels, all_final_preds)
    precision = precision_score(all_labels, all_final_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_final_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_final_preds, average="macro", zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_final_preds)
    sensitivity = (conf_matrix.diagonal() / conf_matrix.sum(axis=1)).mean()
    
    total = conf_matrix.sum()
    specificities = []
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = conf_matrix[i, :].sum() - TP
        FP = conf_matrix[:, i].sum() - TP
        TN = total - TP - FN - FP
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificities.append(spec)

    specificity = np.mean(specificities)

    num_classes = len(set(all_labels))
    if num_classes == 2:
        auc = roc_auc_score(all_labels, [p[1] for p in all_avg_probs])
    else:
        auc = roc_auc_score(all_labels, all_avg_probs, multi_class="ovr", average="macro")

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


def ensemble_predict_soft(models, test_loader, weighted = False, best_model_idx = -1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_final_preds = []
    all_labels = []
    all_avg_probs = []

    if weighted and best_model_idx != -1:
        weights = torch.tensor([1.0 if i != best_model_idx else 1.5 for i in range(len(models))]).view(1, -1, 1)
    else:
        weights = torch.tensor([1.0 for model in models]).view(1, -1, 1)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            batch_probs_per_model = []
            for model in models:
                guesses = model(images)
                probs = F.softmax(guesses, dim=1)

                batch_probs_per_model.append(probs.cpu())

            batch_probs_per_model = torch.stack(batch_probs_per_model, dim=1)

            avg_probs = (batch_probs_per_model * weights).sum(dim=1) / weights.sum()

            final_preds = avg_probs.argmax(dim=1)

            all_avg_probs.append(avg_probs)
            all_final_preds.extend(final_preds.tolist())
            all_labels.extend(labels.tolist())
    
    all_avg_probs = torch.cat(all_avg_probs, dim=0).numpy()
    
    accuracy = accuracy_score(all_labels, all_final_preds)
    precision = precision_score(all_labels, all_final_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_final_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_final_preds, average="macro", zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_final_preds)
    sensitivity = (conf_matrix.diagonal() / conf_matrix.sum(axis=1)).mean()
    
    total = conf_matrix.sum()
    specificities = []
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = conf_matrix[i, :].sum() - TP
        FP = conf_matrix[:, i].sum() - TP
        TN = total - TP - FN - FP
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificities.append(spec)

    specificity = np.mean(specificities)

    num_classes = len(set(all_labels))
    if num_classes == 2:
        auc = roc_auc_score(all_labels, [p[1] for p in all_avg_probs])
    else:
        auc = roc_auc_score(all_labels, all_avg_probs, multi_class="ovr", average="macro")

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
