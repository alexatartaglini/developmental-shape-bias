import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import argparse
import probabilities_to_decision
import helper.human_categories as sc
import math
import random
from scipy import spatial
import os
import json
import numpy as np
import pandas as pd
import glob
from PIL import Image
from data import GeirhosStyleTransferDataset, GeirhosTriplets, FakeStimTrials, calculate_dataset_stats
from plot import plot_class_values, plot_similarity_histograms, plot_norm_histogram, plot_similarity_bar
from evaluate import csv_class_values, calculate_totals, calculate_proportions, calculate_similarity_totals
import clip


def get_penultimate_layer(model, image):
    """Extracts the activations of the penultimate layer for a given input
    image.

    :param model: the model to extract activations from
    :param image: the image to be passed through the model
    :return: the activation of the penultimate layer"""

    layer = model._modules.get('avgpool')
    activations = torch.zeros(2048)

    def copy_data(m, i, o):
        activations.copy_(o.data)

    h = layer.register_forward_hook(copy_data)

    model(image)

    h.remove()

    return activations


def get_embeddings(dir, model, model_type, t, g):
    """ Retrieves embeddings for each image in a dataset from the penultimate
    layer of a given model. Stores the embeddings in a dictionary (indexed by
    image name, eg. cat4-truck3). Returns the dictionary and stores it in a json
    file (model_type_embeddings.json)

    :param dir: path of the dataset
    :param model: the model to extract the embeddings from
    :param model_type: the type of model, eg. saycam, resnet50, etc.
    :param t: true if running triplet simulations
    :param g: true if using grayscale Geirhos dataset

    :return: a dictionary indexed by image name that contains the embeddings for
        all images in a dataset extracted from the penultimate layer of a given
        model.
    """

    try:
        os.mkdir('embeddings')
    except FileExistsError:
        pass

    # Initialize dictionary
    embedding_dict = {}

    # Initialize dataset
    if t:
        dataset = GeirhosStyleTransferDataset(dir, '')
        num_images = dataset.__len__()
        if g:
            embedding_dir = 'embeddings/' + model_type + '_gray.json'
        else:
            embedding_dir = 'embeddings/' + model_type + '_embeddings.json'
    else:
        trials = FakeStimTrials()
        num_images = len(trials.all_stims.keys())
        embedding_dir = 'embeddings/' + model_type + '_fake.json'

    # Remove the final layer from the model
    if model_type == 'saycam' or model_type == 'saycamA' or model_type == 'saycamS'\
            or model_type == 'saycamY':
        modules = list(model.module.children())[:-1]
        penult_model = nn.Sequential(*modules)
    elif model_type == 'resnet50' or model_type == 'mocov2' or model_type == 'swav':
        modules = list(model.children())[:-1]
        penult_model = nn.Sequential(*modules)
    elif model_type == 'clipRN50' or model_type == 'clipViTB32' or model_type == 'dino_resnet50':
        penult_model = model
    elif model_type == 'alexnet' or model_type == 'vgg16':
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        penult_model = model

    for p in penult_model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        # Iterate over images
        for i in range(num_images):
            if t:
                im, name, shape, texture, shape_spec, texture_spec = dataset.__getitem__(i)
            else:
                im, name = trials.getitem(i)
            im = im.unsqueeze(0)

            # Pass image into model
            if model_type == 'clipRN50' or model_type == 'clipViTB32':
                embedding = penult_model.encode_image(im)
            else:
                embedding = penult_model(im).numpy().squeeze()

            embedding_dict[name] = embedding.tolist()

    with open(embedding_dir, 'w') as file:
        json.dump(embedding_dict, file)

    return embedding_dict


def generate_fake_triplets(model_type, model, shape_dir, t, g, f, n=230431):
    """Generates fake embeddings that have the same dimensionality as
     model_type for n triplets, then calculates cosine similarity & dot product
     statistics.

     :param model_type: resnet50, saycam, etc.
     :param n: number of fake triplets to generate. Default is the number of
               triplets the real models see.
     :param t: true if doing triplet simulations
     :param g: true if using grayscale Geirhos dataset
     :param f: true if using artificial dataset
     :param n: number of triplets to generate"""

    # Retrieve embedding magnitude statistics from the real model
    try:
        embeddings = json.load(open('embeddings/' + model_type + '_embeddings.json'))
    except FileNotFoundError:
        embeddings = get_embeddings(shape_dir, model, model_type, t, g)

    avg = 0
    num_embeddings = 0
    min_e = math.inf
    max_e = 0

    size = len(list(embeddings.values())[0])

    for embedding in embeddings.values():
        num_embeddings += 1
        mag = np.linalg.norm(embedding)
        avg += mag
        if mag > max_e:
            max_e = mag
        if mag < min_e:
            min_e = mag

    avg = avg / num_embeddings

    columns = ['Model', 'Anchor', 'Shape Match', 'Texture Match',
               'Shape Dot', 'Shape Cos', 'Texture Dot', 'Texture Cos'
               'Shape Dot Closer', 'Shape Cos Closer', 'Texture Dot Closer', 'Texture Cos Closer']
    results = pd.DataFrame(index=range(n), columns=columns)

    try:
        os.mkdir('results/' + model_type +'/similarity/fake')
    except FileExistsError:
        pass

    # Iterate over n fake triplets
    for t in range(n):
        anchor = []
        shape_match = []
        texture_match = []

        lists = [anchor, shape_match, texture_match]
        new_lists = []

        # Generate three random vectors
        for l in lists:

            for idx in range(size):
                l.append(random.random())

            mag = -1
            while mag < 0:
                mag = np.random.normal(loc=avg, scale=min(avg - min_e, max_e - avg) / 2)

            l = np.array(l)
            current_mag = np.linalg.norm(l)
            new_l = (mag * l) / current_mag
            new_lists.append(new_l)

        anchor = new_lists[0]
        shape_match = new_lists[1]
        texture_match = new_lists[2]

        results.at[t, 'Anchor'] = anchor
        results.at[t, 'Shape Match'] = shape_match
        results.at[t, 'Texture Match'] = texture_match

        shape_dot = np.dot(anchor, shape_match)
        shape_cos = spatial.distance.cosine(anchor, shape_match)
        texture_dot = np.dot(anchor, texture_match)
        texture_cos = spatial.distance.cosine(anchor, texture_match)

        results.at[t, 'Shape Dot'] = shape_dot
        results.at[t, 'Shape Cos'] = shape_cos
        results.at[t, 'Texture Dot'] = texture_dot
        results.at[t, 'Texture Cos'] = texture_cos

        if shape_dot > texture_dot:
            results.at[t, 'Shape Dot Closer'] = 1
            results.at[t, 'Texture Dot Closer'] = 0
        else:
            results.at[t, 'Shape Dot Closer'] = 0
            results.at[t, 'Texture Dot Closer'] = 1

        if shape_cos > texture_cos:
            results.at[t, 'Shape Cos Closer'] = 1
            results.at[t, 'Texture Cos Closer'] = 0
        else:
            results.at[t, 'Shape Cos Closer'] = 0
            results.at[t, 'Texture Cos Closer'] = 1

    results.to_csv('results/' + model_type +'/similarity/fake/fake.csv')
    calculate_similarity_totals(model_type, f, g)


def triplets(model_type, embeddings, verbose, g, shape_dir):
    """First generates all possible triplets of the following form:
    (anchor image, shape match, texture match). Then retrieves the activations
    of the penultimate layer of a given model for each image in the triplet.
    Finally, computes and stores cosine similarity, dot products, Euclidean distances:
    anchor x shape match, anchor x texture match. This determines whether the model
    thinks the shape or texture match for an anchor image is closer to the anchor and
    essentially provides a secondary measure of shape/texture bias.

    :param model_type: resnet50, saycam, etc.
    :param embeddings: a dictionary of embeddings for each image for the given model
    :param verbose: true if results should be printed to the terminal.
    :param g: true if a grayscale version of the Geirhos dataset should be used.
    :param shape_dir: directory for the Geirhos dataset."""

    if g:
        if not os.path.isdir('stimuli-shape/style-transfer-gray'):  # Create grayscale images
            os.mkdir('stimuli-shape/style-transfer-gray')
            for c_path in glob.glob('stimuli-shape/style-transfer/*'):
                category = c_path.split('/')[2]
                os.mkdir('stimuli-shape/style-transfer-gray/' + category)

                for im_path in glob.glob(c_path + '/*.png'):
                    img = Image.open(im_path).convert('LA')
                    im_name = im_path.split('/')[3]
                    img.save('stimuli-shape/style-transfer-gray/' + category + '/' + im_name)

        rgb_mean, rgb_std = calculate_dataset_stats('stimuli-shape/style-transfer-gray', 1, False)
        gray_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=rgb_mean, std=rgb_std)
            ])
        t = GeirhosTriplets(shape_dir, transform=gray_transform)
    else:
        t = GeirhosTriplets(shape_dir)  # Default transforms

    images = t.shape_classes.keys()
    all_triplets = t.triplets_by_image

    sim_dict = {}

    columns = ['Model', 'Anchor', 'Anchor Shape', 'Anchor Texture', 'Shape Match',
               'Texture Match', 'Shape Dot', 'Shape Cos', 'Shape ED',
               'Texture Dot', 'Texture Cos', 'Texture ED', 'Shape Dot Closer',
               'Shape Cos Closer', 'Shape ED Closer',
               'Texture Dot Closer', 'Texture Cos Closer', 'Texture ED Closer']

    cosx = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    for anchor in images:  # Iterate over possible anchor images
        anchor_triplets = all_triplets[anchor]['triplets']
        num_triplets = len(anchor_triplets)

        df = pd.DataFrame(index=range(num_triplets), columns=columns)
        df['Anchor'] = anchor[:-4]
        df['Model'] = model_type
        df['Anchor Shape'] = t.shape_classes[anchor]['shape_spec']
        df['Anchor Texture'] = t.shape_classes[anchor]['texture_spec']

        for i in range(num_triplets):  # Iterate over possible triplets
            triplet = anchor_triplets[i]
            shape_match = triplet[1]
            texture_match = triplet[2]

            df.at[i, 'Shape Match'] = shape_match[:-4]
            df.at[i, 'Texture Match'] = texture_match[:-4]

            # Retrieve images corresponding to names
            # anchor_im, shape_im, texture_im = t.getitem(triplet)

            # Get image embeddings
            anchor_output = torch.FloatTensor(embeddings[anchor])
            shape_output = torch.FloatTensor(embeddings[shape_match])
            texture_output = torch.FloatTensor(embeddings[texture_match])

            if anchor_output.shape == (1,1024) or anchor_output.shape == (1,512):
                anchor_output = torch.squeeze(anchor_output, 0)
                shape_output = torch.squeeze(shape_output, 0)
                texture_output = torch.squeeze(texture_output, 0)

            # Retrieve similarities if they've already been calculated
            if (anchor, shape_match) in sim_dict.keys() or (shape_match, anchor) in sim_dict.keys():
                try:
                    shape_dot = sim_dict[(anchor, shape_match)][0]
                    shape_cos = sim_dict[(anchor, shape_match)][1]
                    shape_ed = sim_dict[(anchor, shape_match)][2]
                except KeyError:
                    shape_dot = sim_dict[(shape_match, anchor)][0]
                    shape_cos = sim_dict[(shape_match, anchor)][1]
                    shape_ed = sim_dict[(shape_match, anchor)][2]
            else:
                shape_dot = np.dot(anchor_output, shape_output)
                shape_cos = cosx(anchor_output, shape_output)
                shape_ed = torch.cdist(torch.unsqueeze(shape_output, 0), torch.unsqueeze(anchor_output, 0))
                sim_dict[(anchor, shape_match)] = [shape_dot, shape_cos, shape_ed]

            if (anchor, texture_match) in sim_dict.keys() or (texture_match, anchor) in sim_dict.keys():
                try:
                    texture_dot = sim_dict[(anchor, texture_match)][0]
                    texture_cos = sim_dict[(anchor, texture_match)][1]
                    texture_ed = sim_dict[(anchor, texture_match)][2]
                except KeyError:
                    texture_dot = sim_dict[(texture_match, anchor)][0]
                    texture_cos = sim_dict[(texture_match, anchor)][1]
                    texture_ed = sim_dict[(texture_match, anchor)][2]
            else:
                texture_dot = np.dot(anchor_output, texture_output)
                texture_cos = cosx(anchor_output, texture_output)
                texture_ed = torch.cdist(torch.unsqueeze(texture_output, 0), torch.unsqueeze(anchor_output, 0))
                sim_dict[(anchor, texture_match)] = [texture_dot, texture_cos, texture_ed]

            if verbose:
                print("For " + anchor + " paired with " + shape_match + ", " + texture_match + ":")
                print("\tShape match dot product: " + str(shape_dot))
                print("\tShape match cos similarity: " + str(shape_cos.item()))
                print("\tShape match Euclidean distance: " + str(shape_ed.item()))
                print("\t-------------")
                print("\tTexture match dot: " + str(texture_dot))
                print("\tTexture match cos similarity: " + str(texture_cos.item()))
                print("\tTexture match Euclidean distance: " + str(texture_ed.item()))
                print()

            df.at[i, 'Shape Dot'] = shape_dot
            df.at[i, 'Shape Cos'] = shape_cos.item()
            df.at[i, 'Shape ED'] = shape_ed.item()
            df.at[i, 'Texture Dot'] = texture_dot
            df.at[i, 'Texture Cos'] = texture_cos.item()
            df.at[i, 'Texture ED'] = texture_ed.item()

            # Compare shape/texture results
            if shape_dot > texture_dot:
                df.at[i, 'Shape Dot Closer'] = 1
                df.at[i, 'Texture Dot Closer'] = 0
            else:
                df.at[i, 'Shape Dot Closer'] = 0
                df.at[i, 'Texture Dot Closer'] = 1

            if shape_cos > texture_cos:
                df.at[i, 'Shape Cos Closer'] = 1
                df.at[i, 'Texture Cos Closer'] = 0
            else:
                df.at[i, 'Shape Cos Closer'] = 0
                df.at[i, 'Texture Cos Closer'] = 1

            if shape_ed < texture_ed:
                df.at[i, 'Shape ED Closer'] = 1
                df.at[i, 'Texture ED Closer'] = 0
            else:
                df.at[i, 'Shape ED Closer'] = 0
                df.at[i, 'Texture ED Closer'] = 1

        if g:
            df.to_csv('results/' + model_type + '/grayscale/' + anchor[:-4] + '.csv', index=False)
        else:
            df.to_csv('results/' + model_type + '/similarity/' + anchor[:-4] + '.csv', index=False)


def fake_stimuli(model_type, embeddings, verbose):

    try:
        os.mkdir('results/' + model_type + '/fake')
    except FileExistsError:
        pass

    trials = FakeStimTrials()
    stimuli = trials.all_stims.keys()
    all_trials = trials.trials_by_image

    columns = ['Model', 'Anchor', 'Anchor Shape', 'Anchor Texture', 'Anchor Color',
               'Shape Match', 'Texture Match', 'Color Match',
               'Shape Cos', 'Texture Cos', 'Color Cos',
               'Shape Dot', 'Texture Dot', 'Color Dot',
               'Shape ED', 'Texture ED', 'Color ED',
               'Shape Cos Closer', 'Texture Cos Closer', 'Color Cos Closer',
               'Shape Dot Closer', 'Texture Dot Closer', 'Color Dot Closer',
               'Shape ED Closer', 'Texture ED Closer', 'Color ED Closer',]

    cosx = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    for anchor in stimuli:  # Iterate over all possible anchors
        anchor_trials = all_trials[anchor]['trials']
        num_trials = len(anchor_trials)

        if num_trials == 0:
            continue

        df = pd.DataFrame(index=range(num_trials), columns=columns)
        df['Anchor'] = anchor
        df['Model'] = model_type
        df['Anchor Shape'] = trials.all_stims[anchor]['shape']
        df['Anchor Texture'] = trials.all_stims[anchor]['texture']
        df['Anchor Color'] = trials.all_stims[anchor]['color']

        for i in range(num_trials):  # Iterate over possible trials
            trial = anchor_trials[i]
            shape_match = trial[1]
            texture_match = trial[2]
            color_match = trial[3]

            df.at[i, 'Shape Match'] = shape_match[:-4]
            df.at[i, 'Texture Match'] = texture_match[:-4]
            df.at[i, 'Color Match'] = color_match[:-4]

            # Get image embeddings
            anchor_output = torch.FloatTensor(embeddings[anchor])
            shape_output = torch.FloatTensor(embeddings[shape_match])
            texture_output = torch.FloatTensor(embeddings[texture_match])
            color_output = torch.FloatTensor(embeddings[color_match])

            if model_type == 'clipRN50' or model_type == 'clipViTB32':
                anchor_output = torch.squeeze(anchor_output, 0)
                shape_output = torch.squeeze(shape_output, 0)
                texture_output = torch.squeeze(texture_output, 0)
                color_output = torch.squeeze(color_output, 0)

            # Cosine similarity
            shape_cos = cosx(anchor_output, shape_output)
            texture_cos = cosx(anchor_output, texture_output)
            color_cos = cosx(anchor_output, color_output)

            # Dot
            shape_dot = np.dot(anchor_output, shape_output)
            texture_dot = np.dot(anchor_output, texture_output)
            color_dot = np.dot(anchor_output, color_output)

            # Euclidean distance
            shape_ed = torch.cdist(torch.unsqueeze(shape_output, 0), torch.unsqueeze(anchor_output, 0))
            texture_ed = torch.cdist(torch.unsqueeze(texture_output, 0), torch.unsqueeze(anchor_output, 0))
            color_ed = torch.cdist(torch.unsqueeze(color_output, 0), torch.unsqueeze(anchor_output, 0))

            df.at[i, 'Shape Cos'] = shape_cos.item()
            df.at[i, 'Texture Cos'] = texture_cos.item()
            df.at[i, 'Color Cos'] = color_cos.item()

            df.at[i, 'Shape Dot'] = shape_dot
            df.at[i, 'Texture Dot'] = texture_dot
            df.at[i, 'Color Dot'] = color_dot

            df.at[i, 'Shape ED'] = shape_ed.item()
            df.at[i, 'Texture ED'] = texture_ed.item()
            df.at[i, 'Color ED'] = color_ed.item()

            if shape_cos > texture_cos and shape_cos > color_cos:
                df.at[i, 'Shape Cos Closer'] = 1
                df.at[i, 'Texture Cos Closer'] = 0
                df.at[i, 'Color Cos Closer'] = 0
            elif texture_cos > shape_cos and texture_cos > color_cos:
                df.at[i, 'Shape Cos Closer'] = 0
                df.at[i, 'Texture Cos Closer'] = 1
                df.at[i, 'Color Cos Closer'] = 0
            else:
                df.at[i, 'Shape Cos Closer'] = 0
                df.at[i, 'Texture Cos Closer'] = 0
                df.at[i, 'Color Cos Closer'] = 1

            if shape_dot > texture_dot and shape_dot > color_dot:
                df.at[i, 'Shape Dot Closer'] = 1
                df.at[i, 'Texture Dot Closer'] = 0
                df.at[i, 'Color Dot Closer'] = 0
            elif texture_dot > shape_dot and texture_dot > color_dot:
                df.at[i, 'Shape Dot Closer'] = 0
                df.at[i, 'Texture Dot Closer'] = 1
                df.at[i, 'Color Dot Closer'] = 0
            else:
                df.at[i, 'Shape Dot Closer'] = 0
                df.at[i, 'Texture Dot Closer'] = 0
                df.at[i, 'Color Dot Closer'] = 1

            if shape_ed < texture_ed and shape_ed < color_ed:
                df.at[i, 'Shape ED Closer'] = 1
                df.at[i, 'Texture ED Closer'] = 0
                df.at[i, 'Color ED Closer'] = 0
            elif texture_ed < shape_ed and texture_ed < color_ed:
                df.at[i, 'Shape ED Closer'] = 0
                df.at[i, 'Texture ED Closer'] = 1
                df.at[i, 'Color ED Closer'] = 0
            else:
                df.at[i, 'Shape ED Closer'] = 0
                df.at[i, 'Texture ED Closer'] = 0
                df.at[i, 'Color ED Closer'] = 1

        df.to_csv('results/' + model_type + '/fake/' + anchor[:-4] + '.csv', index=False)


def initialize_model(model_type):
    """Initializes the model and puts it into evaluation mode. Returns the model.

    :param model_type: resnet50, saycam, etc.

    :return: the loaded model in evaluation mode."""

    if model_type == 'saycam':
        # Load Emin's pretrained SAYCAM model + ImageNet classifier from its .tar file
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        model = nn.DataParallel(model)
        checkpoint = torch.load('models/fz_IN_resnext50_32x4d_augmentation_True_SAY_5_288.tar',
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'mocov2':  # Pre-trained (800 epochs) ResNet-50, MoCoV2
        model = models.resnet50(pretrained=False)
        checkpoint = torch.load('models/moco_v2_800ep_pretrain.pth.tar',
                                map_location=torch.device('cpu'))['state_dict']

        for k in list(checkpoint.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                checkpoint[k[len("module.encoder_q."):]] = checkpoint[k]
            # delete renamed or unused k
            del checkpoint[k]

        model.load_state_dict(checkpoint, strict=False)
    elif model_type == 'saycamA':
        model = models.resnext50_32x4d(pretrained=False)
        model = nn.DataParallel(model)
        checkpoint = torch.load('models/TC-A.tar', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif model_type == 'saycamS':
        model = models.resnext50_32x4d(pretrained=False)
        model = nn.DataParallel(model)
        checkpoint = torch.load('models/TC-S.tar', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif model_type == 'saycamY':
        model = models.resnext50_32x4d(pretrained=False)
        model = nn.DataParallel(model)
        checkpoint = torch.load('models/TC-Y.tar', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_type == 'clipRN50':
        model, _ = clip.load('RN50', device='cpu')
    elif model_type == 'clipViTB32':
        model, _ = clip.load('ViT-B/32', device='cpu')
    elif model_type == 'dino_resnet50':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif model_type == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model_type == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_type == 'swav':
        model = torch.hub.load('facebookresearch/swav', 'resnet50')
    else:
        print('The model ' + model_type + ' has not yet been defined. Please see main.py')
        sys.exit()

    # Put model in evaluation mode
    model.eval()
    return model


def run_simulations(args, model_type):
    """By default: passes images one at a time through a given model and stores/plots the results
    (the shape/texture of the image, the classification made, and whether or not
    the classifcation was a shape classification, a texture classification, or neither.)

    If t flag: runs triplet simulations using the Geirhos dataset
               (see documentation for the triplets function).
    If f flag: runs quadruplet simulations using Brenden Lake's artificial stimulus dataset
               (see documentation for the fake_stimuli function).
    If g flag: runs simulations using a grayscale version of the Geirhos dataset.

    By default, the model is the SAYCAM-trained resnext model. This can be changed when running
    this program in the terminal by using -m 'model_type' or the --all flag, which will run
    desired simulations for all available models.

    :param args: command line arguments
    :param model_type: the type of model, saycam by default. Try -m 'resnet50' to change,
        for example."""

    batch_size = 1
    shape_categories = sc.get_human_object_recognition_categories()  # list of 16 classes in the Geirhos style-transfer dataset
    plot = args.plot
    verbose = args.verbose
    t = args.triplets
    f = args.fake
    g = args.grayscale

    if g:
        shape_dir = 'stimuli-shape/style-transfer-gray'
    else:
        shape_dir = 'stimuli-shape/style-transfer'
    texture_dir = 'stimuli-texture/style-transfer'

    # Initialize the model and put in evaluation mode
    model = initialize_model(model_type)

    # Create directories for results and plots
    try:
        os.mkdir('results/' + model_type)
    except FileExistsError:
        pass

    try:
        os.mkdir('figures/' + model_type)
    except FileExistsError:
        pass

    # Run simulations
    if t:
        if g:
            try:
                os.mkdir('results/' + model_type + '/grayscale')
            except FileExistsError:
                pass

            try:
                embeddings = json.load(open('embeddings/' + model_type + '_gray.json'))
            except FileNotFoundError:
                embeddings = get_embeddings(shape_dir, model, model_type, t, g)
        else:
            try:
                os.mkdir('results/' + model_type + '/similarity')
            except FileExistsError:
                pass

            try:
                embeddings = json.load(open('embeddings/' + model_type + '_embeddings.json'))
            except FileNotFoundError:
                embeddings = get_embeddings(shape_dir, model, model_type, t, g)

        triplets(model_type, embeddings, verbose, g, shape_dir)
        calculate_similarity_totals(model_type, f, g)

        if plot:
            plot_similarity_histograms(model_type, g)
            plot_norm_histogram(model_type, f, g)

    elif f:
        try:
            os.mkdir('results/' + model_type + '/fake')
        except FileExistsError:
            pass

        try:
            os.mkdir('figures/' + model_type + '/fake')
        except FileExistsError:
            pass

        try:
            embeddings = json.load(open('embeddings/' + model_type + '_fake.json'))
        except FileNotFoundError:
            embeddings = get_embeddings('', model, model_type, t, g)

        fake_stimuli(model_type, embeddings, verbose)
        calculate_similarity_totals(model_type, f, g)

    else:
        shape_dict = dict.fromkeys(shape_categories)  # for storing the results
        shape_categories0 = [shape + '0' for shape in shape_categories]
        shape_dict0 = dict.fromkeys(shape_categories0)

        shape_spec_dict = dict.fromkeys(shape_categories)  # contains lists of specific textures for each shape
        for shape in shape_categories:
            shape_dict[shape] = shape_dict0.copy()
            shape_spec_dict[shape] = []

        # Load and process the images using my custom Geirhos style transfer dataset class
        style_transfer_dataset = GeirhosStyleTransferDataset(shape_dir, texture_dir)
        style_transfer_dataloader = DataLoader(style_transfer_dataset, batch_size=1, shuffle=False)
        if not os.path.isdir('stimuli-texture'):
            style_transfer_dataset.create_texture_dir('stimuli-shape/style-transfer', 'stimuli-texture')

        # Obtain ImageNet - Geirhos mapping
        mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
        softmax = nn.Softmax(dim=1)
        softmax2 = nn.Softmax(dim=0)

        with torch.no_grad():
            # Pass images into the model one at a time
            for batch in style_transfer_dataloader:
                im, im_dir, shape, texture, shape_spec, texture_spec = batch

                # hack to extract vars
                im_dir = im_dir[0]
                shape = shape[0]
                texture = texture[0]
                shape_spec = shape_spec[0]
                texture_spec = texture_spec[0]

                output = model(im)
                soft_output = softmax(output).detach().numpy().squeeze()

                decision, class_values = mapping.probabilities_to_decision(soft_output)

                shape_idx = shape_categories.index(shape)
                texture_idx = shape_categories.index(texture)
                if class_values[shape_idx] > class_values[texture_idx]:
                    decision_idx = shape_idx
                else:
                    decision_idx = texture_idx
                decision_restricted = shape_categories[decision_idx]
                restricted_class_values = torch.Tensor([class_values[shape_idx], class_values[texture_idx]])
                restricted_class_values = softmax2(restricted_class_values)

                if verbose:
                    print('Decision for ' + im_dir + ': ' + decision)
                    print('\tRestricted decision: ' + decision_restricted)
                if plot:
                    plot_class_values(shape_categories, class_values, im_dir, shape, texture, model_type)

                shape_dict[shape][texture_spec + '0'] = [decision, class_values,
                                                         decision_restricted, restricted_class_values]
                shape_spec_dict[shape].append(texture_spec)

            csv_class_values(shape_dict, shape_categories, shape_spec_dict, 'results/' + model_type)
            calculate_totals(shape_categories, 'results/' + model_type, verbose)
            calculate_proportions('results/' + model_type, verbose)


if __name__ == '__main__':
    """This file is used to load models, retrieve image embeddings, and run simulations.
    See the documentation for each function above for more information."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Example: saycam, resnet50', required=False, default='saycam')
    parser.add_argument('-v', '--verbose', help='Prints results.', required=False, action='store_true')
    parser.add_argument('-p', '--plot', help='Plots results.', required=False, action='store_true')
    parser.add_argument('-t', '--triplets', help='Obtains similarities for triplets of images.',
                        required=False, action='store_true')
    parser.add_argument('-f', '--fake', help='Obtains similarities for artificial, novel stimuli.',
                        required=False, action='store_true')
    parser.add_argument('-g', '--grayscale', help='Runs simulations with a grayscale version of the Geirhos dataset.',
                        required=False, action='store_true')
    parser.add_argument('--all', help='Generates plots, summaries, or results for all models.', required=False, action='store_true')
    args = parser.parse_args()

    a = args.all
    plot = args.plot

    if a:
        if not plot:
            model_list = ['saycam', 'saycamA', 'saycamS', 'saycamY', 'resnet50', 'clipRN50', 'clipViTB32',
                      'dino_resnet50', 'alexnet', 'vgg16', 'swav', 'mocov2']

            for model_type in model_list:
                print("Running simulations for {0}".format(model_type))
                run_simulations(args, model_type)
        else:
            g = args.grayscale
            f = args.fake
            plot_similarity_bar(g, f)
    else:
        run_simulations(args, args.model)
