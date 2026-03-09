import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
plt.ioff()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def get_model(model_name, num_classes=1):
    if model_name == 'resnet18':
        model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg16_bn':
        model_ft = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_b2':
        model_ft = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)


    elif model_name == 'densenet121':
        model_ft = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    # [TODO - REVISION]
    elif model_name == 'efficientnet_v2_s':
        model_ft = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vit_b_16':
        model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)

    return model_ft

