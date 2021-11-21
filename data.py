import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import shutil
import json
import glob
import warnings
import cv2
from math import inf
import transformers
from random import randint
warnings.filterwarnings("ignore")


def calculate_dataset_stats(path, num_channels, c):
    """This function calculates and returns the mean and std of an image dataset.
    Should be used to determine values for normalization for transforms.

    :param path: the path to the dataset.
    :param num_channels: the number of channels (ie. 1, 3).
    :param c: True if using artificial/cartoon dataset

    :return: two num_channels length lists, one containing the mean and one
             containing the std."""

    if c:
        classes = ['']
    else:
        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    pixel_num = 0
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)

    for idx, d in enumerate(classes):
        im_paths = glob.glob(os.path.join(path, d, "*.png"))
        for path in im_paths:
            im = cv2.imread(path)   # image in M*N*CHANNEL_NUM shape, channel in BGR order
            im = im / 255.0
            pixel_num += (im.size / num_channels)
            channel_sum = channel_sum + np.sum(im, axis=(0, 1))
            channel_sum_squared = channel_sum_squared + np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std


class GeirhosStyleTransferDataset(Dataset):
    """A custom Dataset class for the Geirhos Style Transfer dataset."""

    def __init__(self, shape_dir, texture_dir, transform):
        """
        :param shape_dir: a directory for the style transfer dataset organized by shape
        :param texture_dir: a directory for the style transfer dataset organized by texture
        :param transform: a set of image transformations. NOTE: if the model being used is
                          ViTB16, this will actually be the feature extractor object, which
                          performs transforms on images for this model.
        """

        self.shape_dir = shape_dir
        self.texture_dir = texture_dir
        self.shape_classes = {}
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

        if self.transform:
            if type(self.transform) == transformers.models.vit.feature_extraction_vit.ViTFeatureExtractor:
                image = self.transform(images=image, return_tensors="pt")
            else:
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

    def __init__(self, transform, shape_dir='stimuli-shape/style-transfer', same_instance=True):
        """Generates/loads the triplets. all_triplets is a list of all 3-tuples.
        triplets_by_image is a dictionary; the keys are image names, and it stores all
        shape/texture matches plus all possible triplets for a given image (as the anchor).

        :param transform: a set of image transformations. NOTE: if the model being used is
                          ViTB16, this will actually be the feature extractor object, which
                          performs transforms on images for this model.
        :param shape_dir: directory for the Geirhos dataset.
        :param same_instance: true if shape/texture matches should be by specific instance -
                              ie. cat4 is a match for cat4 but not for cat1. When false,
                              shape/texture matches are exclusively different instances -
                              ie. cat4 is a match for cat1, but cat4 is not a match for
                              cat4.
        """

        self.shape_classes = {}
        self.all_triplets = []
        self.triplets_by_image = {}
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
            if same_instance:
                self.triplets_by_image = json.load(open('geirhos_triplets.json'))
            else:
                self.triplets_by_image = json.load(open('geirhos_triplets_different_instances.json'))
            self.all_triplets = self.triplets_by_image['all']
            self.triplets_by_image.pop('all')

        except FileNotFoundError:
            # Generate triplets
            for image in self.shape_classes.keys(): # Iterate over anchor images
                shape = self.shape_classes[image]['shape']
                texture = self.shape_classes[image]['texture']
                shape_spec = self.shape_classes[image]['shape_spec']
                texture_spec = self.shape_classes[image]['texture_spec']

                if not same_instance:
                    shape_spec1 = shape_spec
                    texture_spec1 = texture_spec
                    shape_spec = shape
                    texture_spec = texture

                self.triplets_by_image[image] = {}
                self.triplets_by_image[image]['shape matches'] = []
                self.triplets_by_image[image]['texture matches'] = []
                self.triplets_by_image[image]['triplets'] = []

                for shape_match in glob.glob(shape_dir + '/' + shape + '/' + shape_spec + '*.png'):
                    if shape_match.split('/')[3].split('-')[0] != shape_spec:
                        continue
                    shape_match = shape_match.split('/')[-1]
                    if shape_match == image or shape_match not in self.shape_classes.keys():
                        continue
                    elif self.shape_classes[shape_match]['texture'] == texture:
                        continue  # Filters shape matches with same texture class
                    if not same_instance:
                        if shape_spec1 == self.shape_classes[shape_match]['shape_spec']:
                            continue
                    self.triplets_by_image[image]['shape matches'].append(shape_match)

                for texture_match in glob.glob(shape_dir + '/*/*' + texture_spec + '*.png'):
                    if texture_match.split('/')[3].split('-')[1][:-4] != texture_spec:
                        continue
                    texture_match = texture_match.split('/')[-1]
                    if texture_match == image or texture_match not in self.shape_classes.keys():
                        continue
                    elif self.shape_classes[texture_match]['shape'] == shape:
                        continue  # Filter out texture matches with same shape class
                    if not same_instance:
                        if texture_spec1 == self.shape_classes[texture_match]['texture_spec']:
                            continue
                    self.triplets_by_image[image]['texture matches'].append(texture_match)

                for shape_match in self.triplets_by_image[image]['shape matches']:
                    for texture_match in self.triplets_by_image[image]['texture matches']:
                        triplet = [image, shape_match, texture_match]
                        self.triplets_by_image[image]['triplets'].append(triplet)
                        self.all_triplets.append(triplet)

            self.triplets_by_image['all'] = self.all_triplets

            # Save dictionary as a JSON file
            if same_instance:
                triplet_dir = 'geirhos_triplets.json'
            else:
                triplet_dir = 'geirhos_triplets_different_instances.json'
            with open(triplet_dir, 'w') as file:
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
            if type(self.transform) == transformers.models.vit.feature_extraction_vit.ViTFeatureExtractor:
                ims = [anchor_im, shape_im, texture_im]
                outs = {key: None for key in ims}
                for im in ims:
                    outs[im] = self.transform(images=im, return_tensors="pt")
                return outs[0], outs[1], outs[2]

            else:
                anchor_im = self.transform(anchor_im)
                shape_im = self.transform(shape_im)
                texture_im = self.transform(texture_im)

        return anchor_im.unsqueeze(0), shape_im.unsqueeze(0), texture_im.unsqueeze(0)

    def max_num_triplets(self):
        """This method returns the minimum number of triplets that an anchor image
        in the triplet dataset has. This minimum number should be treated as a
        maximum number of triplets allowed for a given anchor to be included in
        the results; this way, all images are equally represented as anchors. This
        ensures that certain images are not overrepresented as anchors.

        :return: a cap for the number of triplets that should be allowed for any given
                 anchor."""

        min_triplets = inf

        for anchor in self.triplets_by_image.keys():
            num_triplets = len(self.triplets_by_image[anchor]['triplets'])

            if num_triplets < min_triplets:
                min_triplets = num_triplets

        return min_triplets

    def select_capped_triplets(self, draw):
        """This method randomly selects min_triplets (see max_num_triplets) triplets for
        each anchor image and inserts these into a dictionary indexed by anchor image.
        This allows one to evaluate a set of triplets such that no anchor image is
        overrepresented.

        :param draw: the random draw to choose from the seed.json file (ensures all
                     models view the same random selection)

        :return: a dictionary of randomly selected triplets per anchor image such that
                 each anchor image has an equal number of triplets."""

        selections = json.load(open('seed.json'))
        return selections[str(draw)]


class CartoonStimTrials(Dataset):
    """This class provides a way to generate and access all possible trials of novel,
    artificial stimuli from the stimuli-shape/cartoon directory. Each stimulus consists of
    a shape, color, and texture. A trial consists of an anchor image, a shape match, a
    color match, and a texture match."""

    def __init__(self, transform, cartoon_dir='stimuli-shape/cartoon'):
        """Generates/loads all possible trials. all_trials is a list of all 4-tuples.
        trials_by_image is a dictionary; the keys are the image paths, and it stores
        all shape/color/texture matches plus all possible trials for the given image
        as the anchor.

        :param transform: appropriate transforms for the given model (should match training
                      data stats). NOTE: if the model being used is ViTB16, this will actually
                      be the feature extractor object, which performs transforms on images for
                      this model.
        :param cartoon_dir: directory for artificial/cartoon images
        :param transform: transforms to be applied"""

        self.all_stims = {}  # Contains shape, texture, & color classifications for all images
        self.all_trials = []
        self.trials_by_image = {}
        self.transform = transform

        # Create/load dictionaries containing shape/texture/color classifications for each image
        try:
            # Load dictionaries
            self.all_stims = json.load(open('cartoon_stimulus_classes.json'))  # Dictionary of dictionaries

        except FileNotFoundError:

            # Create dictionaries
            for image_dir in glob.glob(cartoon_dir + '/*.png'):
                image = image_dir.split('/')[2]  # file name
                specs = image.replace('.png', '').split('_')  # [shape, texture, color]

                if 'x' in specs or 'X' in specs:
                    os.remove(image_dir)
                    continue
                else:
                    self.all_stims[image] = {}
                    self.all_stims[image]['shape'] = specs[0]
                    self.all_stims[image]['texture'] = specs[1]
                    self.all_stims[image]['color'] = specs[2]
                    self.all_stims[image]['dir'] = image_dir

            # Save dictionary as a JSON file
            with open('cartoon_stimulus_classes.json', 'w') as file:
                json.dump(self.all_stims, file)

        # Generate/load trials
        try:
            # Load trials
            self.trials_by_image = json.load(open('cartoon_trials.json'))
            self.all_trials = self.trials_by_image['all']
            self.trials_by_image.pop('all')

        except FileNotFoundError:

            # Generate trials
            for image in self.all_stims.keys():  # Iterate over anchor images
                shape = self.all_stims[image]['shape']
                texture = self.all_stims[image]['texture']
                color = self.all_stims[image]['color']

                self.trials_by_image[image] = {}
                self.trials_by_image[image]['shape matches'] = []
                self.trials_by_image[image]['texture matches'] = []
                self.trials_by_image[image]['color matches'] = []
                self.trials_by_image[image]['trials'] = []

                # Find shape/texture/color matches
                for shape_match in self.all_stims.keys():
                    shape2 = self.all_stims[shape_match]['shape']
                    texture2 = self.all_stims[shape_match]['texture']
                    color2 = self.all_stims[shape_match]['color']
                    if shape_match == image or shape != shape2:  # Same image or different shape
                        continue
                    elif texture == texture2 or color == color2:  # Same texture or color
                        continue
                    self.trials_by_image[image]['shape matches'].append(shape_match)
                for texture_match in self.all_stims.keys():
                    shape2 = self.all_stims[texture_match]['shape']
                    texture2 = self.all_stims[texture_match]['texture']
                    color2 = self.all_stims[texture_match]['color']
                    if texture_match == image or texture != texture2:
                        continue
                    elif shape == shape2 or color == color2:
                        continue
                    self.trials_by_image[image]['texture matches'].append(texture_match)
                for color_match in self.all_stims.keys():
                    shape2 = self.all_stims[color_match]['shape']
                    texture2 = self.all_stims[color_match]['texture']
                    color2 = self.all_stims[color_match]['color']
                    if color_match == image or color != color2:
                        continue
                    elif shape == shape2 or texture == texture2:
                        continue
                    self.trials_by_image[image]['color matches'].append(color_match)

                # Create trials
                for shape_match in self.trials_by_image[image]['shape matches']:
                    for texture_match in self.trials_by_image[image]['texture matches']:
                        if texture_match == shape_match:
                            continue
                        for color_match in self.trials_by_image[image]['color matches']:
                            if color_match == texture_match or color_match == shape_match:
                                continue
                            trial = [image, shape_match, texture_match, color_match]
                            self.trials_by_image[image]['trials'].append(trial)
                            self.all_trials.append(trial)

            self.trials_by_image['all'] = self.all_trials

            # Save dictionary as a JSON file
            with open('cartoon_trials.json', 'w') as file:
                json.dump(self.trials_by_image, file)

    def make_square(self, im, min_size=224, fill_color=(255, 255, 255)):
        """This function pads rectangular images with white space.

        :param im: input image
        :param min_size: the minimum size that an image should be (224x224 by default)
        :param fill_color: by default, white"""

        """        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGBA', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))"""

        x, y = im.size
        size = max(min_size, x, y)
        im.load()  # needed for split()
        new_im = Image.new('RGB', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)), mask=im.split()[3])
        return new_im

    def __len__(self):
        return len(self.trials_by_image.keys())

    def __getitem__(self, trial):
        """For a given singular index, returns the singular image corresponding to that index.

        :param trial: a singular integer index.
        :return: the image with transforms applied and the image name."""

        name = list(self.all_stims.keys())[trial]
        path = self.all_stims[name]['dir']

        im = Image.open(path)
        im = self.make_square(im)

        if self.transform:
            if type(self.transform) == transformers.models.vit.feature_extraction_vit.ViTFeatureExtractor:
                im = self.transform(images=im, return_tensors="pt")
            else:
                im = self.transform(im)

        return im, name

    def select_capped_triplets(self, draw):
        """Currently just returns all quadruplets."""
        return self.all_trials

    def max_num_triplets(self):
        """Currently just returns the total number of quadruplets."""
        return len(self.all_trials)


class SilhouetteTriplets:
    """This class is a variant of the GeirhosTriplets class. It is almost identical
    to this class, except it generates stimuli that consist of the original Geirhos
    style transfer images overlaid with the silhouettes of the shape classes.

    The result is stimuli that are composed of the silhouette of one object and
    filled with the texture of another. This is an attempt to make texture less
    salient and devise a more accurate measure of shape bias."""

    def __init__(self, transform, alpha, novel=False, bg=None, shape_dir='stimuli-shape/style-transfer'):
        """Generates/loads the triplets. all_triplets is a list of all 3-tuples.
        triplets_by_image is a dictionary; the keys are image names, and it stores all
        shape/texture matches plus all possible triplets for a given image (as the anchor).

        :param transform: a set of image transformations. NOTE: if the model being used is
                          ViTB16, this will actually be the feature extractor object, which
                          performs transforms on images for this model.
        :param shape_dir: directory for the Geirhos dataset.
        :param alpha: controls transparency of masks. 1 = fully opaque masks. 0 = original
                      Geirhos stimuli.
        :param novel: set to True if using novel stimulus shapes; note that alpha=1.0 in
                      this case.
        :param bg: (optional) the path to an image to be used as a background for silhouette
                   stimuli (for the alpha=1.0 case). Replaces the solid white background of
                   the stimuli with the image located at the given path.
        :param s_dir: the directory where the shape silhouettes are located.
        """

        self.shape_classes = {}
        self.all_triplets = []
        self.triplets_by_image = {}
        self.transform = transform
        self.bg = bg
        self.novel = novel

        if self.bg or self.novel:
            alpha = 1.0

        self.alpha = int(alpha * 255)
        self.alpha_str = str(alpha)

        # Create/load dictionaries containing shape and texture classifications for each image
        if self.novel:
            shape_dir = 'stimuli-shape/novel-silhouettes-brodatz-' + str(self.alpha / 255)

            try:
                # Load dictionary
                self.shape_classes = json.load(open('novel_shape_classes.json'))

            except FileNotFoundError:
                shapes = [os.path.basename(x)[:-4] for x in glob.glob('stimuli-shape/novel-masks/*')]
                textures = [os.path.basename(x)[:-4] for x in glob.glob('stimuli-shape/brodatz-textures/*')]

                for shape in shapes:
                    for texture in textures:
                        stimulus = shape + '-' + texture + '.png'

                        self.shape_classes[stimulus] = {'shape': shape, 'texture': texture,
                                                   'dir': shape_dir + '/' + shape + '/' + stimulus}

                with open('novel_shape_classes.json', 'w') as file:
                    json.dump(self.shape_classes, file)

            # Generate/load triplets
            try:
                # Load triplets
                self.triplets_by_image = json.load(open('novel_triplets.json'))
                self.all_triplets = self.triplets_by_image['all']
                self.triplets_by_image.pop('all')

            except FileNotFoundError:
                for image in self.shape_classes.keys():  # Iterate over anchor images
                    shape = self.shape_classes[image]['shape']
                    texture = self.shape_classes[image]['texture']

                    self.triplets_by_image[image] = {}
                    self.triplets_by_image[image]['shape matches'] = []
                    self.triplets_by_image[image]['texture matches'] = []
                    self.triplets_by_image[image]['triplets'] = []

                    for potential_match in self.shape_classes.keys():
                        if potential_match == image:
                            continue
                        elif self.shape_classes[potential_match]['shape'] == shape:
                            self.triplets_by_image[image]['shape matches'].append(potential_match)
                        elif self.shape_classes[potential_match]['texture'] == texture:
                            self.triplets_by_image[image]['texture matches'].append(potential_match)

                    for shape_match in self.triplets_by_image[image]['shape matches']:
                        for texture_match in self.triplets_by_image[image]['texture matches']:
                            triplet = [image, shape_match, texture_match]
                            self.triplets_by_image[image]['triplets'].append(triplet)
                            self.all_triplets.append(triplet)

                self.triplets_by_image['all'] = self.all_triplets

                # Save dictionary as a JSON file
                triplet_dir = 'novel_triplets.json'
                with open(triplet_dir, 'w') as file:
                    json.dump(self.triplets_by_image, file)

            self.create_silhouette_stimuli(alpha=self.alpha, s_dir='stimuli-shape/novel-masks')
        else:
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
                for image in self.shape_classes.keys(): # Iterate over anchor images
                    shape = self.shape_classes[image]['shape']
                    texture = self.shape_classes[image]['texture']
                    shape_spec = self.shape_classes[image]['shape_spec']
                    texture_spec = self.shape_classes[image]['texture_spec']

                    self.triplets_by_image[image] = {}
                    self.triplets_by_image[image]['shape matches'] = []
                    self.triplets_by_image[image]['texture matches'] = []
                    self.triplets_by_image[image]['triplets'] = []

                    for shape_match in glob.glob(shape_dir + '/' + shape + '/' + shape_spec + '*.png'):
                        if shape_match.split('/')[3].split('-')[0] != shape_spec:
                            continue
                        shape_match = shape_match.split('/')[-1]
                        if shape_match == image or shape_match not in self.shape_classes.keys():
                            continue
                        elif self.shape_classes[shape_match]['texture'] == texture:
                            continue  # Filters shape matches with same texture class
                        self.triplets_by_image[image]['shape matches'].append(shape_match)

                    for texture_match in glob.glob(shape_dir + '/*/*' + texture_spec + '*.png'):
                        if texture_match.split('/')[3].split('-')[1][:-4] != texture_spec:
                            continue
                        texture_match = texture_match.split('/')[-1]
                        if texture_match == image or texture_match not in self.shape_classes.keys():
                            continue
                        elif self.shape_classes[texture_match]['shape'] == shape:
                            continue  # Filter out texture matches with same shape class
                        self.triplets_by_image[image]['texture matches'].append(texture_match)

                    for shape_match in self.triplets_by_image[image]['shape matches']:
                        for texture_match in self.triplets_by_image[image]['texture matches']:
                            triplet = [image, shape_match, texture_match]
                            self.triplets_by_image[image]['triplets'].append(triplet)
                            self.all_triplets.append(triplet)

                self.triplets_by_image['all'] = self.all_triplets

                # Save dictionary as a JSON file
                triplet_dir = 'geirhos_triplets.json'
                with open(triplet_dir, 'w') as file:
                    json.dump(self.triplets_by_image, file)

            self.create_silhouette_stimuli(alpha=self.alpha)

    def create_silhouette_stimuli(self, alpha, s_dir='stimuli-shape/filled-silhouettes'):
        """Create and save the silhouette stimuli if they do not already exist.

        :param alpha: controls the transparency of the masks. alpha = 0 generates the
                      original Geirhos dataset. alpha = 1 generates a dataset with fully
                      opaque masks. Any alpha in (0, 1) will mask the background texture to
                      varying degrees (with white pixels).
        :param s_dir: the path to a folder containing the shape masks. By default, uses
                      the Geirhos filled silhouettes as masks. Setting novel=True when
                      initializing a SilhouetteTriplets object means that novel shape
                      masks will be used instead."""

        alpha_dir = '-' + str(alpha / 255)

        if self.novel:
            try:
                os.mkdir('stimuli-shape/novel-silhouettes-brodatz' + alpha_dir)
            except FileExistsError:
                return

            for im_name in self.shape_classes.keys():
                im_path = self.shape_classes[im_name]['dir']
                mask_path = s_dir + '/' + self.shape_classes[im_name]['shape'] + '.png'

                try:
                    os.mkdir('stimuli-shape/novel-silhouettes-brodatz' + alpha_dir + '/' +
                             self.shape_classes[im_name]['shape'])
                except FileExistsError:
                    pass

                im = Image.open('stimuli-shape/brodatz-textures' + '/' + self.shape_classes[im_name]['texture'] + '.png')
                mask = Image.open(mask_path).convert('RGBA')
                mask_data = mask.getdata()

                bound = im.size[0] - mask.size[0]
                x = randint(0, bound)
                y = randint(0, bound)
                im = im.crop((x, y, x + mask.size[0], y + mask.size[0]))

                new_data = []
                for item in mask_data:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append(item)
                    else:
                        new_data.append((0, 0, 0, 255 - alpha))

                mask.putdata(new_data)

                base = Image.new('RGB', mask.size, (255, 255, 255))
                base.paste(im, mask=mask.split()[3])

                base = base.resize((180, 180))
                base2 = Image.new('RGB', mask.size, (255, 255, 255))

                img_w, img_h = base.size
                bg_w, bg_h = base2.size
                offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)

                base2.paste(base, offset)
                base2.save(im_path)
        else:
            try:
                os.mkdir('stimuli-shape/texture-silhouettes' + alpha_dir)
            except FileExistsError:
                return

            for im_name in self.shape_classes.keys():
                im_path = self.shape_classes[im_name]['dir']
                mask_path = s_dir + '/' + self.shape_classes[im_name]['shape'] + '/' + \
                            self.shape_classes[im_name]['shape_spec'] + '.png'

                try:
                    os.mkdir('stimuli-shape/texture-silhouettes' + alpha_dir + '/' +
                             self.shape_classes[im_name]['shape'])
                except FileExistsError:
                    pass

                im = Image.open(im_path)
                mask = Image.open(mask_path).convert('RGBA')
                mask_data = mask.getdata()

                new_data = []
                for item in mask_data:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append(item)
                    else:
                        new_data.append((0, 0, 0, 255 - alpha))

                mask.putdata(new_data)

                base = Image.new('RGB', mask.size, (255, 255, 255))
                base.paste(im, mask=mask.split()[3])
                base.save('stimuli-shape/texture-silhouettes' + alpha_dir + '/' +
                          self.shape_classes[im_name]['shape'] + '/' + im_name)

    def __getitem__(self, idx):
        """For a given singular index, returns the singular image corresponding to that index.

        :param idx: a singular integer index.
        :return: the image with transforms applied and the image name."""

        name = list(self.shape_classes.keys())[idx]
        if self.novel:
            path = 'stimuli-shape/novel-silhouettes-brodatz-1.0/' + name
        else:
            path = 'stimuli-shape/texture-silhouettes-' + str(self.alpha / 255) + '/' + self.shape_classes[name]['shape'] \
               + '/' + name
        im = Image.open(path)

        if self.transform:
            if type(self.transform) == transformers.models.vit.feature_extraction_vit.ViTFeatureExtractor:
                im = self.transform(images=im, return_tensors="pt")
            else:
                im = self.transform(im)

        return im, name

    def __len__(self):
        return len(self.shape_classes.keys())

    def getitem(self, triplet):
        """For a given (anchor, shape match, texture match) triplet, loads and returns
        all 3 images. Not to be confused with __getitem__.

        :param triplet: a length-3 list containing the name of an anchor, shape match,
            and texture match.

        :return: the anchor, shape match, and texture match images with transforms applied."""

        if self.novel:
            s_dir = 'stimuli-shape/novel-silhouettes-brodatz-1.0/'
        else:
            s_dir = 'stimuli-shape/texture-silhouettes-' + str(self.alpha / 255) + '/'

        anchor_path = s_dir + self.shape_classes[triplet[0]]['shape'] + '/' + triplet[0]
        shape_path = s_dir + self.shape_classes[triplet[1]]['shape'] + '/' + triplet[1]
        texture_path = s_dir + self.shape_classes[triplet[2]]['shape'] + '/' + triplet[2]

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

    def max_num_triplets(self):
        """This method returns the minimum number of triplets that an anchor image
        in the triplet dataset has. This minimum number should be treated as a
        maximum number of triplets allowed for a given anchor to be included in
        the results; this way, all images are equally represented as anchors. This
        ensures that certain images are not overrepresented as anchors.

        :return: a cap for the number of triplets that should be allowed for any given
                 anchor."""

        min_triplets = inf

        for anchor in self.triplets_by_image.keys():
            num_triplets = len(self.triplets_by_image[anchor]['triplets'])

            if num_triplets < min_triplets:
                min_triplets = num_triplets

        return min_triplets

    def select_capped_triplets(self, draw):
        """This method randomly selects min_triplets (see max_num_triplets) triplets for
        each anchor image and inserts these into a dictionary indexed by anchor image.
        This allows one to evaluate a set of triplets such that no anchor image is
        overrepresented.

        :param draw: the random draw to choose from the seed.json file (ensures all
                     models view the same random selection)

        :return: a dictionary of randomly selected triplets per anchor image such that
                 each anchor image has an equal number of triplets."""

        if self.novel:
            selections = json.load(open('novel_seed.json'))
        else:
            selections = json.load(open('seed.json'))

        return selections[str(draw)]

    def get_alpha_str(self):
        return self.alpha_str
