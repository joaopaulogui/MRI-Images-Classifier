
import torch
from src.services.transforms import get_test_transforms
from src.services.dataset import get_data_loader
from src.services.models.squeezenet import setup_squeezenet
from src.services.models.densenet import setup_densenet
from src.services.models.resnet import setup_resnet
from src.services.models.metrics import evaluate_model

import torch
from torchvision import datasets

def evaluate(data_dir, num_workers):

    ds = datasets.ImageFolder(root=data_dir)

    test_transforms = get_test_transforms(224, 224)
    test_loader = get_data_loader(ds, test_transforms, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "DenseNet": setup_densenet,
        "ResNet": setup_resnet,
        "SqueezeNet": setup_squeezenet,
    }

    for name, setup_fn in models.items():
        _eval_model(name, f"generated/{name.lower()}.pth", setup_fn, test_loader, device)
        _eval_model(name+" With KFold", f"generated/{name.lower()}-with-kfold.pth", setup_fn, test_loader, device)

    

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