import h5py
import numpy as np
from PIL import Image


def explore(nome, obj, nivel=0):
    indent = "  " * nivel
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 {nome}/")
        for chave in obj.keys():
            explore(chave, obj[chave], nivel + 1)
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 {nome} — shape: {obj.shape}, dtype: {obj.dtype}")

for i in range(1, 3065):
    with h5py.File(f'resources/Figshare/mats/{i}.mat', 'r') as f:
        image = f['cjdata']['image'][:]
        label = f['cjdata']['label'][()].flat[0]

        if image.ndim == 2:
            image = image.T
        elif image.ndim == 3:
            image = np.transpose(image, (2, 1, 0))

        image = image.astype(np.float32)

        p_low  = np.percentile(image, 1)
        p_high = np.percentile(image, 99)
        image = np.clip(image, p_low, p_high)

        image = (image - p_low) / (p_high - p_low) * 255

        img = Image.fromarray(image.astype(np.uint8))

        match label:
            case 1:
                img.save(f'resources/Figshare/meningioma_tumor/{i}.jpg')
            case 2:
                img.save(f'resources/Figshare/glioma_tumor/{i}.jpg')
            case 3:
                img.save(f'resources/Figshare/pituitary_tumor/{i}.jpg')
