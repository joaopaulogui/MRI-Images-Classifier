
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.services.models.metrics import evaluate_model

def evaluate(test_loader, model_name, setup_fn):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics, classes = _eval_model(model_name, f"generated/{model_name.lower()}.pth", setup_fn, test_loader, device)
    k_fold_metrics, kfold_classes = _eval_model(model_name+" With KFold", f"generated/{model_name.lower()}-with-kfold.pth", setup_fn, test_loader, device)

    _plot_confusion_matrix(metrics["conf_matrix"], classes, f"{model_name}-confusion-matrix.png", f"Matriz de confusão do modelo {model_name}")
    
    _plot_confusion_matrix(k_fold_metrics["conf_matrix"], kfold_classes, f"{model_name}-kfold-confusion-matrix.png", f"Matriz de confusão do modelo {model_name} com Kfold")

    disp = ConfusionMatrixDisplay(confusion_matrix=metrics["conf_matrix"])
    disp.plot()
    plt.title(f"Matriz de confusão do modelo {model_name}")
    plt.savefig(f"generated/graphs/{model_name}-confusion-matrix.png")
    
    kfold_disp = ConfusionMatrixDisplay(confusion_matrix=k_fold_metrics["conf_matrix"])
    kfold_disp.plot()
    plt.title(f"Matriz de confusão do modelo {model_name} com Kfold")
    plt.savefig(f"generated/graphs/{model_name}-kfold-confusion-matrix.png")

    return {
        "model_metrics": metrics,
        "kfold_model_metrics": k_fold_metrics,
    }
    
    

def _eval_model(model_name, checkpoint_dir, setup_fn, test_loader, device):
    checkpoint = torch.load(checkpoint_dir, map_location=device)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = setup_fn(device, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    metrics = evaluate_model(model, test_loader)
    print(f"{model_name} val AUC: {metrics['auc']*100:.2f}% | " 
                f"accuracy: {metrics['accuracy']*100:.2f}% | "
                f"precision: {metrics['precision']*100:.2f}% | "
                f"recall: {metrics['recall']*100:.2f}% | "
                f"f1-score: {metrics['f1']*100:.2f}% | "
                f"sensitivity: {metrics['sensitivity']*100:.2f}% | "
                f"specificity: {metrics['specificity']*100:.2f}%")
    
    return metrics, classes

def _plot_confusion_matrix(cm, class_names, file_name, title, figsize=(8, 8)):
    n = len(class_names)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    ax.invert_yaxis()
    ax.axis("off")

    # Células principais da matriz
    for i in range(n):
        for j in range(n):
            valor = cm[i, j]
            pct = valor / total * 100
            cor = "#b1dcc0" if i == j else "#ffc7bf"  # verde na diagonal, vermelho fora
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cor, edgecolor="white"))
            ax.text(j + 0.5, i + 0.4, f"{valor}", ha="center", va="center", fontweight="bold")
            ax.text(j + 0.5, i + 0.65, f"{pct:.1f}%", ha="center", va="center", fontsize=9)

    # Coluna de recall (à direita)
    for i in range(n):
        acertos = cm[i, i]
        total_linha = cm[i, :].sum()
        recall = acertos / total_linha * 100 if total_linha > 0 else 0
        erro = 100 - recall
        ax.add_patch(plt.Rectangle((n, i), 1, 1, facecolor="#efefef", edgecolor="white"))
        ax.text(n + 0.5, i + 0.4, f"{recall:.1f}%", ha="center", va="center", color="green", fontweight="bold")
        ax.text(n + 0.5, i + 0.65, f"{erro:.1f}%", ha="center", va="center", color="red", fontsize=9)

    # Linha de precisão (embaixo)
    for j in range(n):
        acertos = cm[j, j]
        total_coluna = cm[:, j].sum()
        precision = acertos / total_coluna * 100 if total_coluna > 0 else 0
        erro = 100 - precision
        ax.add_patch(plt.Rectangle((j, n), 1, 1, facecolor="#efefef", edgecolor="white"))
        ax.text(j + 0.5, n + 0.4, f"{precision:.1f}%", ha="center", va="center", color="green", fontweight="bold")
        ax.text(j + 0.5, n + 0.65, f"{erro:.1f}%", ha="center", va="center", color="red", fontsize=9)

    # Célula do canto (acurácia total)
    acc_total = np.trace(cm) / total * 100
    erro_total = 100 - acc_total
    ax.add_patch(plt.Rectangle((n, n), 1, 1, facecolor="#d6d6d6", edgecolor="black", linewidth=0.5))
    ax.text(n + 0.5, n + 0.4, f"{acc_total:.1f}%", ha="center", va="center", color="green", fontweight="bold")
    ax.text(n + 0.5, n + 0.65, f"{erro_total:.1f}%", ha="center", va="center", color="red", fontsize=9)

    filtered_class_names = [name.replace("_tumor", "") for name in class_names if name != "no_tumor"]

    # Labels dos eixos
    for i, nome in enumerate(filtered_class_names):
        ax.text(-0.1, i + 0.5, nome, ha="right", va="center", fontweight="bold")
        ax.text(i + 0.5, n + 1.15, nome, ha="center", va="center", fontweight="bold")

    ax.text((n) / 2, -0.3, "Predicted Class", ha="center", fontsize=11)
    fig.text(0.02, 0.5, "True Class", va="center", rotation="vertical", fontsize=11)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"generated/graphs/{file_name}", dpi=150, bbox_inches="tight")
