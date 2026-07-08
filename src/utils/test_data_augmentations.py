from torchvision import datasets
from PIL import Image
import os
import numpy as np
import albumentations as A

training_transforms = A.Compose([

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.Rotate(limit=5, border_mode=0, p=0.5),

        A.Affine(
            shear={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            p=0.4,
        ),
        A.Downscale(scale_range=(0.85, 0.95), p=0.3),

        A.GaussianBlur(blur_limit=(3, 5), p=0.3),

        A.GaussNoise(std_range=(0.001, 0.015), p=0.3),

        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.1, 0.2), p=0.4),
        A.RandomGamma(gamma_limit=(90, 120), p=0.3),

        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.Resize(224, 224),
    ])

testing_transforms = A.Compose([
        A.Resize(224, 224),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
])

def test_training_augmentation():

    dataset = datasets.ImageFolder(root='resources/Kaggle/Training')

    for idx, (img, label) in enumerate(dataset):
        classe = dataset.classes[label]
        pasta_saida = f'generated/data_augmented_images/{classe}'
        os.makedirs(pasta_saida, exist_ok=True)

        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = training_transforms(image=img_np)['image']
        img_np = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_np).save(f'generated/data_augmented_images/{classe}/{idx}.jpg')

def test_testing_augmentation():

    dataset = datasets.ImageFolder(root='generated/data_augmented_images')
    for idx, (img, label) in enumerate(dataset):
        pasta_saida = f'generated/data_augmented_images/testing'
        os.makedirs(pasta_saida, exist_ok=True)

        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = testing_transforms(image=img_np)['image']
        img_np = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_np).save(f'generated/data_augmented_images/testing/{idx}.jpg')

def test_data_augmentation():
    test_training_augmentation()
    test_testing_augmentation()