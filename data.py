import os
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import json
import glob
import warnings
from math import inf
import transformers
from random import randint
warnings.filterwarnings("ignore")


sizes = [(45, 45), (90, 90), (135, 135), (180, 180), (224, 224)]
percents = ['20', '40', '60', '80', '100']


class SilhouetteTriplets(Dataset):
    """This class provides a way to generate and access all possible triplets of
    stimuli. These triplets consist of an anchor image (eg. cat4-truck3.png),
    a shape match to the anchor image (eg. cat4-boat2.png), and a texture match to
    the anchor (eg. dog3-truck3.png).

    The shape and texture matches are specific: ie., cat4-truck3.png is a shape match
    for cat4-knife2.png but not for cat2-knife2.png.

    The purpose of these triplets is to measure similarity between shape matches/texture
    matches and the anchor image after having been passed through a model.

    This class can also generate stimuli with any alpha, size, or background setting.
    """

    def __init__(self, args, stimuli_dir, transform, displace_bg=False, override=False,
                 num_triplets='max'):
        """Generates/loads the triplets. all_triplets is a list of all 3-tuples.
        triplets_by_image is a dictionary; the keys are image names, and it stores all
        shape/texture matches plus all possible triplets for a given image (as the anchor).

        :param args: command line arguments
        :param stimuli_dir: location of stimuli
        :param transform: a set of image transformations. NOTE: if the model being used is
                          ViTB16, this will actually be the feature extractor object, which
                          performs transforms on images for this model.
        :param displace_bg: randomly select a patch of the background image to serve as the
                            stimulus background. If False, the entire image is used.
        :param override: if True, will replace the stimuli that already exist.
        :param num_triplets: if max, use maximum number of possible triplets per anchor such
                             that each anchor is equally represented (see max_num_triplets).
                             if int < max_num_triplets, the number of triplets per anchor is
                             capped at this.
        """
        self.shape_classes = {}
        self.all_triplets = []
        self.triplets_by_image = {}
        self.transform = transform
        self.bg = args.bg
        self.novel = args.novel
        self.alpha = int(args.alpha * 255)
        if args.alpha == 1:
            self.alpha_str = '1'
        else:
            self.alpha_str = str(args.alpha)
        self.blur = args.blur
        if self.blur == 0:
            self.blur_str = ''
        else:
            self.blur_str = '_{0}'.format(str(self.blur))
        self.stimuli_dir = stimuli_dir
        self.percent = args.percent_size
        self.stimulus_size = sizes[percents.index(self.percent)]
        self.unaligned = args.unaligned
        if self.bg and ('brodatz' in self.bg or 'saycam' in self.bg):
            self.displace_bg = True
        else:
            self.displace_bg = displace_bg
        self.override = override
        self.num_triplets = num_triplets

        # Create/load dictionaries containing shape and texture classifications for each image
        if self.novel:
            shape_class_dir = 'shape_classes/novel_shape_classes.json'

            try:
                # Load dictionary
                self.shape_classes = json.load(open(shape_class_dir))

            except FileNotFoundError:
                shapes = [os.path.basename(x)[:-4] for x in glob.glob('stimuli/novel-masks/*')]
                textures = [os.path.basename(x)[:-4] for x in glob.glob('stimuli/brodatz-textures/*')]

                for shape in shapes:
                    for texture in textures:
                        stimulus = '{0}-{1}.png'.format(shape, texture)

                        self.shape_classes[stimulus] = {'shape': shape, 'texture': texture,
                                                   'dir': '{0}/{1}'.format(shape, stimulus)}

                with open(shape_class_dir, 'w') as file:
                    json.dump(self.shape_classes, file)

            # Generate/load triplets
            triplet_dir = 'novel_triplets.json'

            try:
                # Load triplets
                self.triplets_by_image = json.load(open(triplet_dir))
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
                with open(triplet_dir, 'w') as file:
                    json.dump(self.triplets_by_image, file)

            self.create_silhouette_stimuli()

        else:
            shape_class_dir = 'shape_classes/geirhos_shape_classes.json'

            try:
                # Load dictionary
                self.shape_classes = json.load(open(shape_class_dir))

            except FileNotFoundError:
                # Create dictionary
                for image_dir in glob.glob('stimuli/geirhos-alpha0.0-size100-aligned/*/*.png'):
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
                        self.shape_classes[image[3]]['dir'] = '{0}/{1}'.format(shape, image[3])

                # Save dictionary as a JSON file
                with open(shape_class_dir, 'w') as file:
                    json.dump(self.shape_classes, file)

            # Generate/load triplets
            triplet_dir = 'geirhos_triplets.json'

            try:
                # Load triplets
                self.triplets_by_image = json.load(open(triplet_dir))
                self.all_triplets = self.triplets_by_image['all']
                self.triplets_by_image.pop('all')

            except FileNotFoundError:
                # Generate triplets
                for image in self.shape_classes.keys(): # Iterate over anchor images
                    shape_spec = self.shape_classes[image]['shape_spec']
                    texture_spec = self.shape_classes[image]['texture_spec']

                    self.triplets_by_image[image] = {}
                    self.triplets_by_image[image]['shape matches'] = []
                    self.triplets_by_image[image]['texture matches'] = []
                    self.triplets_by_image[image]['triplets'] = []

                    for potential_match in self.shape_classes.keys():
                        if potential_match == image:
                            continue
                        elif self.shape_classes[potential_match]['shape_spec'] == shape_spec:
                            if self.shape_classes[potential_match]['texture_spec'] != texture_spec:
                                self.triplets_by_image[image]['shape matches'].append(potential_match)
                        elif self.shape_classes[potential_match]['texture_spec'] == texture_spec:
                            self.triplets_by_image[image]['texture matches'].append(potential_match)

                    for shape_match in self.triplets_by_image[image]['shape matches']:
                        for texture_match in self.triplets_by_image[image]['texture matches']:
                            triplet = [image, shape_match, texture_match]
                            self.triplets_by_image[image]['triplets'].append(triplet)
                            self.all_triplets.append(triplet)

                self.triplets_by_image['all'] = self.all_triplets

                # Save dictionary as a JSON file
                with open(triplet_dir, 'w') as file:
                    json.dump(self.triplets_by_image, file)

            self.create_silhouette_stimuli()

    def create_silhouette_stimuli(self):
        """Create and save the silhouette stimuli if they do not already exist."""
        try:
            os.mkdir('stimuli/{0}'.format(self.stimuli_dir))
        except FileExistsError:
            if self.override:
                pass
            else:
                return

        if self.novel:
            mask_dir = 'stimuli/novel-masks'
        else:
            mask_dir = 'stimuli/geirhos-masks'

        for im_name in self.shape_classes.keys():

            try:
                os.mkdir('stimuli/{0}/{1}'.format(self.stimuli_dir,
                                                  self.shape_classes[im_name]['shape']))
            except FileExistsError:
                pass

            im_path = 'stimuli/{0}/{1}'.format(self.stimuli_dir,
                                               self.shape_classes[im_name]['dir'])
            if self.novel:
                mask_path = '{0}/{1}.png'.format(mask_dir, self.shape_classes[im_name]['shape'])
            else:
                mask_path = '{0}/{1}/{2}.png'.format(mask_dir, self.shape_classes[im_name]['shape'],
                                                     self.shape_classes[im_name]['shape_spec'])

            mask = Image.open(mask_path).convert('RGBA')

            if self.bg:
                if self.displace_bg:
                    bg = Image.open(self.bg).convert('RGBA').resize((mask.size[0] * 2, mask.size[0] * 2), Image.NEAREST)
                    x = randint(0, 224)
                    y = randint(0, 224)
                    bg = bg.crop((x, y, x + mask.size[0], y + mask.size[0])).filter(ImageFilter.SHARPEN)
                else:
                    bg = Image.open(self.bg).convert('RGBA').resize((224, 224), Image.NEAREST)
                bg.putalpha(255 - self.alpha)

                if self.blur != 0:
                    bg = bg.filter(ImageFilter.GaussianBlur(radius=self.blur))
            else:
                bg = Image.new('RGBA', (224, 224), (255, 255, 255, 255-self.alpha))

            if self.novel:
                texture_path = 'stimuli/brodatz-textures/{0}.png'.format(self.shape_classes[im_name]['texture'])
                texture = Image.open(texture_path).resize((mask.size[0] * 2, mask.size[0] * 2))

                # Attain a randomly selected patch of texture
                bound = texture.size[0] - mask.size[0]
                x = randint(0, bound)
                y = randint(0, bound)
                texture = texture.crop((x, y, x + mask.size[0], y + mask.size[0]))

                # Place mask over texture
                mask_data = mask.getdata()

                new_data = []
                for item in mask_data:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append(item)
                    else:
                        new_data.append((0, 0, 0, 0))

                mask.putdata(new_data)

                base = Image.new('RGBA', mask.size, (255, 255, 255, 0))
                base.paste(texture, mask=mask.split()[3])

                # Resize stimulus if necessary
                if self.percent != '100':
                    resized = base.resize(self.stimulus_size, Image.NEAREST)
                    base = Image.new('RGBA', mask.size, (255, 255, 255, 255))

                    img_w, img_h = resized.size
                    bg_w, bg_h = base.size
                    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)

                    base.paste(resized, offset)
                    base = base.crop((offset[0], offset[1],
                                      offset[0] + self.stimulus_size[0],
                                      offset[1] + self.stimulus_size[0]))
                    '''
                    mask = base

                    width, height = mask.size

                    x = (width - self.stimulus_size[0]) // 2
                    y = (height - self.stimulus_size[0]) // 2

                    mask = mask.crop((x, y, x + self.stimulus_size[0], y + self.stimulus_size[0]))
                    '''

                if self.unaligned:
                    bound = bg.size[0] - base.size[0]
                    x = randint(0, bound)  # not shape aligned when uncommented
                    y = randint(0, bound)  # not shape aligned when uncommented
                    bg.paste(base.convert('RGB'), (x, y), mask=base)
                else:
                    x = (224 - self.stimulus_size[0]) // 2  # shape aligned if uncommented
                    bg.paste(base.convert('RGB'), (x, x), mask=base)

                bg.save(im_path)

            else:
                # Masks are placed first, then resizing is done afterwards
                mask_data = mask.getdata()

                new_data = []
                for item in mask_data:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append(item)
                    else:
                        new_data.append((0, 0, 0, 0))

                mask.putdata(new_data)

                if not self.bg:
                    texture_path = 'stimuli/geirhos-alpha0.0-size100-aligned/{0}'.format(
                        self.shape_classes[im_name]['dir'])
                else:
                    texture_path = 'stimuli/geirhos-alpha1-size100-aligned/{0}'.format(
                        self.shape_classes[im_name]['dir'])
                texture = Image.open(texture_path)

                base = Image.new('RGBA', mask.size, (255, 255, 255, 0))
                base.paste(texture, mask=mask.split()[3])

                # Resize
                if self.percent != '100':
                    base = base.resize(self.stimulus_size)
                if self.unaligned:
                    bound = bg.size[0] - self.stimulus_size[0]
                    x = randint(0, bound)
                    y = randint(0, bound)
                    bg.paste(base.convert('RGB'), (x, y), mask=base)
                else:
                    x = (224 - self.stimulus_size[0]) // 2  # shape aligned if uncommented
                    bg.paste(base.convert('RGB'), (x, x), mask=base)

                bg.save(im_path)

    def __getitem__(self, idx):
        """For a given singular index, returns the singular image corresponding to that index.

        :param idx: a singular integer index.
        :return: the image with transforms applied and the image name."""

        name = list(self.shape_classes.keys())[idx]
        path = 'stimuli/{0}/{1}'.format(self.stimuli_dir, self.shape_classes[name]['dir'])

        im = Image.open(path).convert('RGB')

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

        s_dir = 'stimuli/{0}'.format(self.stimuli_dir)

        anchor_path = '{0}/{1}/{2}'.format(s_dir, self.shape_classes[triplet[0]]['shape'], triplet[0])
        shape_path = '{0}/{1}/{2}'.format(s_dir, self.shape_classes[triplet[1]]['shape'], triplet[1])
        texture_path = '{0}/{1}/{2}'.format(s_dir, self.shape_classes[triplet[2]]['shape'], triplet[2])

        # Load images
        anchor_im = Image.open(anchor_path).convert('RGB')
        shape_im = Image.open(shape_path).convert('RGB')
        texture_im = Image.open(texture_path).convert('RGB')

        # Apply transforms
        if self.transform:
            if type(self.transform) == transformers.models.vit.feature_extraction_vit.ViTFeatureExtractor:
                anchor_im = self.transform(images=anchor_im, return_tensors="pt")
                anchor_im['pixel_values'] = anchor_im['pixel_values'].unsqueeze(0)

                shape_im = self.transform(images=shape_im, return_tensors="pt")
                shape_im['pixel_values'] = shape_im['pixel_values'].unsqueeze(0)

                texture_im = self.transform(images=texture_im, return_tensors="pt")
                texture_im['pixel_values'] = texture_im['pixel_values'].unsqueeze(0)
            else:
                anchor_im = self.transform(anchor_im)
                anchor_im = anchor_im.unsqueeze(0)

                shape_im = self.transform(shape_im)
                shape_im = shape_im.unsqueeze(0)

                texture_im = self.transform(texture_im)
                texture_im = texture_im.unsqueeze(0)

        return anchor_im, shape_im, texture_im

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
        """This method randomly selects min_triplets (see max_num_triplets) triplets or
        num_triplets < min_triplets for each anchor image and inserts these into a
        dictionary indexed by anchor image. This allows one to evaluate a set of triplets
        such that no anchor image is overrepresented.

        :param draw: the random draw to choose from the seed.json file (ensures all
                     models view the same random selection)

        :return: a dictionary of randomly selected triplets per anchor image such that
                 each anchor image has an equal number of triplets."""

        if self.novel:
            selections = json.load(open('novel_seed{}.json'.format(self.num_triplets)))
        else:
            selections = json.load(open('seed{}.json'.format(self.num_triplets)))

        return selections[str(draw)]

    def get_alpha_str(self):
        return self.alpha_str
