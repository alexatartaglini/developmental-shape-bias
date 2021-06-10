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


class GeirhosTriplets:
    """This class provides a way to generate and access all possible triplets of
    Geirhos images. These triplets consist of an anchor image (eg. cat4-truck3.png),
    a shape match to the anchor image (eg. cat4-boat2.png), and a texture match to
    the anchor (eg. dog3-truck3.png).

    The shape and texture matches are specific: ie., cat4-truck3.png is a shape match
    for cat4-knife2.png but not for cat2-knife2.png.

    The purpose of these triplets is to measure similarity between shape matches/texture
    matches and the anchor image after having been passed through a model."""

    def __init__(self, shape_dir, transform=None):
        """Generates/loads the triplets. all_triplets is a list of all 3-tuples.
        triplets_by_image is a dictionary; the keys are image names, and it stores all
        shape/texture matches plus all possible triplets for a given image (as the anchor).

        :param shape_dir: directory for the Geirhos dataset.
        :param transform: a set of image transformations (optional)
        """

        self.shape_classes = {}
        self.all_triplets = []
        self.triplets_by_image = {}

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

        # Generate/load triplets
        try:
            # Load triplets
            self.triplets_by_image = json.load(open('geirhos_triplets.json'))
            self.all_triplets = self.triplets_by_image['all']
            self.triplets_by_image.pop('all')

        except FileNotFoundError:
            # Generate triplets
            self.all_triplets = []

            for image in self.shape_classes.keys(): # Iterate over anchor images
                shape = self.shape_classes[image]['shape']
                shape_spec = self.shape_classes[image]['shape_spec']
                texture_spec = self.shape_classes[image]['texture_spec']

                self.triplets_by_image[image] = {}
                self.triplets_by_image[image]['shape matches'] = []
                self.triplets_by_image[image]['texture matches'] = []
                self.triplets_by_image[image]['triplets'] = []

                for shape_match in glob.glob(shape_dir + '/' + shape + '/' + shape_spec + '*.png'):
                    shape_match = shape_match.split('/')[-1]
                    if shape_match == image or shape_match not in self.shape_classes.keys():
                        continue
                    self.triplets_by_image[image]['shape matches'].append(shape_match)
                for texture_match in glob.glob(shape_dir + '/*/*' + texture_spec + '.png'):
                    texture_match = texture_match.split('/')[-1]
                    if texture_match == image or texture_match not in self.shape_classes.keys():
                        continue
                    self.triplets_by_image[image]['texture matches'].append(texture_match)

                for shape_match in self.triplets_by_image[image]['shape matches']:
                    for texture_match in self.triplets_by_image[image]['texture matches']:
                        triplet = [image, shape_match, texture_match]
                        self.triplets_by_image[image]['triplets'].append(triplet)
                        self.all_triplets.append(triplet)

            self.triplets_by_image['all'] = self.all_triplets

            # Save dictionary as a JSON file
            with open('geirhos_triplets.json', 'w') as file:
                json.dump(self.triplets_by_image, file)

    def getitem(self, triplet):
        """For a given (anchor, shape match, texture match) triplet, loads and returns
        all 3 images.

        :param triplet: a length-3 list containing the name of an anchor, shape match,
            and texture match.
        :return: the anchor, shape match, and texture match images with transforms applied."""

        anchor_path = self.shape_classes[triplet[0]]['dir']
        shape_path = self.shape_classes[triplet[1]]['dir']
        texture_path = self.shape_classes[triplet[2]]['dir']

        # Load images
        anchor_im = Image.open(anchor_path)
        shape_im = Image.open(shape_path)
        texture_im = Image.open(texture_path)

        # Apply transforms
        if self.transform:
            anchor_im = self.transform(anchor_im)
            shape_im = self.transform(shape_im)
            texture_im = self.transform(texture_im)

        return anchor_im.unsqueeze(0), shape_im.unsqueeze(0), texture_im.unsqueeze(0)
