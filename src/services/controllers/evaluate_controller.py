
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.services.models.metrics import evaluate_model

def evaluate(test_loader, model_name, setup_fn):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = _eval_model(model_name, f"generated/{model_name.lower()}.pth", setup_fn, test_loader, device)
    k_fold_metrics = _eval_model(model_name+" With KFold", f"generated/{model_name.lower()}-with-kfold.pth", setup_fn, test_loader, device)

    disp = ConfusionMatrixDisplay(confusion_matrix=metrics["conf_matrix"])
    os.makedirs(os.path.dirname("generated/graphs/"), exist_ok=True)
    disp.plot()
    plt.savefig(f"generated/graphs/{model_name}-confusion-matrix.png")
    
    kfold_disp = ConfusionMatrixDisplay(confusion_matrix=k_fold_metrics["conf_matrix"])
    kfold_disp.plot()
    plt.savefig(f"generated/graphs/{model_name}-kfold-confusion-matrix.png")
    
    

def _eval_model(model_name, checkpoint_dir, setup_fn, test_loader, device):
    checkpoint = torch.load(checkpoint_dir, map_location=device)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = setup_fn(device, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    metrics = evaluate_model(model, test_loader)
    print(f"{model_name} val accuracy: {metrics['accuracy']*100:.2f}% | "
                f"precision: {metrics['precision']*100:.2f}% | "
                f"f1: {metrics['f1']*100:.2f}% | "
                f"sensitivity: {metrics['sensitivity']*100:.2f}% | "
                f"specificity: {metrics['specificity']*100:.2f}%")
    
    return metrics