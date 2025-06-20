import os
import torchvision
import torch.nn as nn
from torchsummary import summary

import configuration as cfg
from custom_dataset import get_classes_name


def get_model(num_classes):
    model = torchvision.models.resnet50(weights="DEFAULT")

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    class_name = get_classes_name(os.path.join(cfg.BASE_PATH, "classname.txt"))

    model = get_model(len(class_name)).to(cfg.DEVICE)
    summary(model, (3, 224, 224))