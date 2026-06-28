import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_width, img_height):
    return A.Compose([
        A.Resize(224, 224),

        # ── Geometria ─────────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Rotação conservadora como o artigo sugere
        A.Rotate(limit=5, border_mode=0, p=0.5),

        # Shear reduzido — ±0.5 rad (~28°) era agressivo demais,
        # ±0.15 rad (~8°) simula os cortes reais sem distorcer anatomia
        A.Affine(
            shear={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            fit_output=True,
            p=0.4,
        ),

        #A.RandomCrop(height=200, width=200, p=0.2),

        # ── Degradação de qualidade ───────────────────────────────────────
        # Downscale moderado — simula scanner antigo sem destruir features
        #A.Downscale(scale_range=(0.85, 0.95), p=0.3),

        # Blur leve a moderado
        #A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # Ruído moderado
        #A.GaussNoise(std_range=(0.001, 0.015), p=0.3),

        # ── Intensidade e contraste ───────────────────────────────────────
        #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.1, 0.2), p=0.4),
        #A.RandomGamma(gamma_limit=(90, 120), p=0.3),

        A.Resize(224, 224),
        A.CLAHE(clip_limit=(2.0, 4.0), tile_grid_size=(8, 8), p=0.5),

        # ── Normalização ImageNet ─────────────────────────────────────────
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_test_transforms(img_width, img_height):
    return A.Compose([
        A.Resize(img_width, img_height),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])