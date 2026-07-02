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
    

    names = []
    accuracies = []
    accuracies_kfold = []

    for name, setup_fn in models.items():
        metrics = evaluate(test_loader, name, setup_fn)
        names.append(name)
        accuracies.append(metrics["model_metrics"]["accuracy"])
        accuracies_kfold.append(metrics["kfold_model_metrics"]["accuracy"])

    plt.figure(figsize=(8,5))
    bars = plt.bar(names, accuracies)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{acc:.2%}", ha="center", va="bottom")

    plt.ylim(0, 1)
    plt.ylabel("Acurácia")
    plt.title("Comparação da acurácia dos modelos")
    plt.savefig("generated/graphs/accuracy-comparison.png")


    kfold_bars = plt.bar(names, accuracies_kfold)

    for bar, acc in zip(kfold_bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height + 0.005, f"{acc:.2%}", ha="center", va="bottom")

    plt.ylim(0, 1)
    plt.ylabel("Acurácia")
    plt.title("Comparação da acurácia dos modelos com kfold")
    plt.savefig("generated/graphs/kfold-accuracy-comparison.png")
