import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


class GeirhosStyleTransferDataset(Dataset):
    """A custom Dataset class for the Geirhos Style Transfer dataset."""

    def __init__(self, shape_dir, texture_dir, transform=None):
        """
        :param shape_dir: a directory for the style transfer dataset organized by shape
        :param texture_dir: a directory for the style transfer dataset organized by texture
        :param transform: a set of image transformations (optional)
        """

        self.shape_dir = shape_dir
        self.texture_dir = texture_dir
        self.shape_classes = {}
        self.texture_classes = {}

        # Default image processing
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Create dictionaries containing shape and texture classifications for each image
        for category in sorted(os.listdir(shape_dir)):
            if category != '.DS_Store':
                for image in sorted(os.listdir(shape_dir + '/' + category)):
                    self.shape_classes[image] = category
                for image in sorted(os.listdir(texture_dir + '/' + category)):
                    self.texture_classes[image] = category

    def __len__(self):
        """
        :return: the number of images in the style transfer dataset.
        """

        return len(self.shape_classes.keys()) # Number of images

    def __getitem__(self, idx):
        """
        :param idx: the index of the image to be accessed
        :return: a tuple with the name of the idx_th image, its shape category, its texture
            category, and the image itself with transforms applied
        """

        images = [key for key in self.shape_classes.keys()]
        image_dir = self.shape_dir + '/' + self.shape_classes[images[idx]] + '/' + images[idx]
        image = Image.open(image_dir)

        if self.transform:
            image = self.transform(image)

        shape = self.shape_classes[images[idx]]
        texture = self.texture_classes[images[idx]]

        return images[idx], shape, texture, image