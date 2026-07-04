import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets

from src.services.transforms import get_test_transforms
from src.services.dataset import get_data_loader
from src.services.models.squeezenet import setup_squeezenet
from src.services.models.densenet import setup_densenet
from src.services.models.resnet import setup_resnet
from src.services.controllers.evaluate_controller import evaluate

def evaluate_pipeline(data_dir, num_workers):

    ds = datasets.ImageFolder(root=data_dir)

    test_transforms = get_test_transforms(224, 224)
    test_loader = get_data_loader(ds, test_transforms, num_workers=num_workers)
    
    models = {
        "DenseNet": setup_densenet,
        "ResNet": setup_resnet,
        "SqueezeNet": setup_squeezenet,
    }

    os.makedirs(os.path.dirname("generated/graphs/"), exist_ok=True)
    

    all_names = []
    all_metrics = {}
    all_metrics_kfold = {}

    for name, setup_fn in models.items():
        metrics = evaluate(test_loader, name, setup_fn)
        all_names.append(name)

        for metric, value in metrics["model_metrics"].items():
            if all_metrics.get(metric) is None:
                all_metrics[metric] = []

            all_metrics[metric].append(value)

        for metric, value in metrics["kfold_model_metrics"].items():
            if all_metrics_kfold.get(metric) is None:
                all_metrics_kfold[metric] = []

            all_metrics_kfold[metric].append(value)
        
    

    plt.figure(figsize=(8,5))

    _save_graph(all_names, all_metrics["accuracy"], "Acurácia", "Comparação da acurácia dos modelos", "accuracy-comparison.png")
    
    _save_graph(all_names, all_metrics_kfold["accuracy"], "Acurácia", "Comparação da acurácia dos modelos com kfold", "kfold_accuracy-comparison.png")
    

def _save_graph(names, metrics, label, title, file_name):
    plt.clf()

    bars = plt.bar(names, metrics)

    for bar, metric in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{metric:.2%}", ha="center", va="bottom")

    plt.ylim(0, 1.1)
    plt.ylabel(label)
    plt.title(title)
    plt.savefig(f"generated/graphs/{file_name}")