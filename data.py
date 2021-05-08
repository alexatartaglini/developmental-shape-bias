import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import shutil
import json
import glob

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

        # Default image processing
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Create/load dictionaries containing shape and texture classifications for each image
        try:
            # Load dictionary
            self.shape_classes = json.load(open('geirhos_shape_classes.json'))

        except FileNotFoundError:
            # Create dictionary
            for image_dir in glob.glob('stimuli-shape/style-transfer/*/*.png'):
                image = image_dir.split('/')
                shape = image[2]  # Shape class of image
                texture_spec = image[3].split('-')[1].replace('.png', '')  # Specific texture instance, eg. clock2
                shape_spec = image[3].split('-')[0]  # Specific shape instance, eg. airplane1
                texture = ''.join([i for i in texture_spec if not i.isdigit()])  # Texture class

                if shape != texture:  # Filter images that are not cue-conflict
                    self.shape_classes[image[3]] = {}  # Initialize dictionary for single image
                    self.shape_classes[image[3]]['shape'] = shape
                    self.shape_classes[image[3]]['texture'] = texture
                    self.shape_classes[image[3]]['shape_spec'] = shape_spec
                    self.shape_classes[image[3]]['texture_spec'] = texture_spec
                    self.shape_classes[image[3]]['dir'] = image_dir

            # Save dictionary as a JSON file
            with open('geirhos_shape_classes.json', 'w') as file:
                json.dump(self.shape_classes, file)

    def __len__(self):
        """
        :return: the number of images in the style transfer dataset.
        """

        return len(self.shape_classes.keys()) # Number of images

    def __getitem__(self, idx):
        """
        :param idx: the index of the image to be accessed
        :return: a tuple with the idx_th image itself with transforms applied, the name of the
        idx_th image, its shape category, its texture category, its specific shape,
        specific texture.
        """

        images = sorted([key for key in self.shape_classes.keys()])

        image_name = images[idx]  # Name of PNG file
        im_dict = self.shape_classes[image_name]  # Dictionary with properties for this image
        image_dir = im_dict['dir']  # Full path to image file

        image = Image.open(image_dir)  # Load image

        #image.show()

        if self.transform:
            image = self.transform(image)

        shape = im_dict['shape']
        texture = im_dict['texture']
        shape_spec = im_dict['shape_spec']
        texture_spec = im_dict['texture_spec']

        return image, image_name, shape, texture, shape_spec, texture_spec

    def create_texture_dir(self, shape_dir, texture_dir):
        """Takes a dataset that is organized by shape category and copies
        images into folders organized by texture category.

        :param shape_dir: the directory of the shape-based dataset.
        :param texture_dir: name of the directory for the texture-based version."""

        texture_path = texture_dir + '/' + shape_dir.split('/')[1]

        try:
            shutil.rmtree(texture_dir)
            os.mkdir(texture_dir)
            os.mkdir(texture_path)
        except:
            os.mkdir(texture_dir)
            os.mkdir(texture_path)

        for category in sorted(os.listdir(shape_dir)):
            if category != '.DS_Store':
                os.mkdir(texture_path + '/' + category)

        for category in sorted(os.listdir(shape_dir)):
            if category != '.DS_Store':
                for image in sorted(os.listdir(shape_dir + '/' + category)):
                    texture = image.replace('.png','').split('-')[1]
                    texture = ''.join([i for i in texture if not i.isdigit()])

                    shutil.copyfile(shape_dir + '/' + category + '/' + image, texture_path + '/' + texture + '/' + image)
