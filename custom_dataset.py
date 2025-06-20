import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import configuration as cfg


def get_transform(train):
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)


def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = image_tensor * std + mean
    return image_tensor.clamp(0, 1)


def get_classes_name(path):
    class_name = []
    with open(path, "r") as f:
        for line in f.readlines():
            class_name.append(line.replace("_", " ").replace("\n", ""))
    return class_name


class CustomDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        sample_row = self.csv.iloc[idx]

        image = Image.open(os.path.join(cfg.BASE_PATH, sample_row["image:FILE"])).convert('RGB')
        label = sample_row["category"]

        if self.transforms:
            image = self.transforms(image)

        return image, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    custom_ds = CustomDataset(
        csv_path=os.path.join(cfg.BASE_PATH, "train.csv"),
        transforms=get_transform(train=True)
    )

    class_name = get_classes_name(os.path.join(cfg.BASE_PATH, "classname.txt"))

    image, label = custom_ds[0]
    print(image.shape)
    image_denormalize = denormalize(image)
    image_denormalize = image_denormalize.numpy().transpose((1, 2, 0))

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Before denormalization")

    axes[1].imshow(image_denormalize)
    axes[1].set_title("After denormalization")

    fig.suptitle(f"Fruit name: {class_name[label]}")
    plt.show()