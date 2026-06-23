
import torch
import cv2
from src.services.transforms import get_test_transforms
from src.services.models.squeezenet import setup_squeezenet

import torch

def infere(image_path):
    transform = get_test_transforms(224, 224)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image)
    tensor = augmented["image"]
    tensor = tensor.unsqueeze(0)
    tensor = tensor.float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("generated/squeezenet.pth")

    classes = checkpoint["classes"]
    num_classes = len(classes)

    squeezenet = setup_squeezenet(device, num_classes)

    squeezenet.load_state_dict(checkpoint["model_state_dict"])

    squeezenet.eval()

    with torch.no_grad():
        guess = squeezenet(tensor)

    print(guess)


    #TODO: eval and vote system