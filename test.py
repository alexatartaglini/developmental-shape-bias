import torch
import torchvision.models as models
from torchvision import datasets, transforms, models
import PIL
import copy
import os
import numpy as np
import shutil
import probabilities_to_decision
import matplotlib.pyplot as plt

def transform_data(path, batch):
    # Using more or less the same preprocessing as Emin (but without center crop)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_datasets = datasets.ImageFolder(path, preprocess)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch,
                                                    shuffle=True)
    dataset_sizes = len(image_datasets)

    return dataloaders, dataset_sizes


if __name__ == '__main__':

    batch_size = 4

    # Load Emin's pretrained SAYCAM model from the .tar file. I'm using more or less
    # the same code that Emin uses in imagenet_finetuning.py
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=2765, bias=True)
    model = torch.nn.DataParallel(model) # What does this do?
    checkpoint = torch.load('models/resnext50_S.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Add an ImageNet classifier. Not sure if I'm doing all of this correctly
    model.module.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)

    # Process the images
    images, size = transform_data('stimuli/style-transfer/', batch_size)

    # Obtain ImageNet - Geirhos mapping
    mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
    softmax = torch.nn.Softmax(dim=1)

    # Pass the images through the model in batches of 4
    for i, (image_batch, target) in enumerate(images):
        outputs = model(image_batch) # 4 1000-length tensors
        soft_outputs = softmax(outputs)
        decisions = []

        for j in range(batch_size):
            output = soft_outputs[j].detach().numpy()
            decisions.append(mapping.probabilities_to_decision(output))

        print(decisions)
        print(target)

