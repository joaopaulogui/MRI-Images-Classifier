from torchvision import datasets
from PIL import Image
import os
import numpy as np
import albumentations as A

transforms = A.Compose([
        A.Resize(224, 224),

        # ── Geometria ─────────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),

        # Rotação conservadora como o artigo sugere
        A.Rotate(limit=5, border_mode=0, p=0.5),

        # Shear reduzido — ±0.5 rad (~28°) era agressivo demais,
        # ±0.15 rad (~8°) simula os cortes reais sem distorcer anatomia
        A.Affine(
            shear={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            fit_output=True,
            p=0.4,
        ),

        A.RandomCrop(height=200, width=200, p=0.2),

        # ── Degradação de qualidade ───────────────────────────────────────
        # Downscale moderado — simula scanner antigo sem destruir features
        A.Downscale(scale_range=(0.85, 0.95), p=0.3),

        # Blur leve a moderado
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # Ruído moderado
        A.GaussNoise(std_range=(0.001, 0.015), p=0.3),

        # ── Intensidade e contraste ───────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.1, 0.2), p=0.5),
        A.RandomGamma(gamma_limit=(85, 120), p=0.3),

        A.Resize(224, 224),
        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.5),
    ])


def test():

    dataset = datasets.ImageFolder(root='resources/Kaggle/Training')

    for idx, (img, label) in enumerate(dataset):
        classe = dataset.classes[label]
        pasta_saida = f'generated/data_augmented_images/{classe}'
        os.makedirs(pasta_saida, exist_ok=True)

        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = transforms(image=img_np)['image']
        img_np = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_np).save(f'generated/data_augmented_images/{classe}/{idx}.jpg')