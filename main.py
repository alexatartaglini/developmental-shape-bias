import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import argparse
import math
import random
from scipy import spatial
import os
import json
import numpy as np
import pandas as pd
import glob
from PIL import Image
from data import GeirhosStyleTransferDataset, GeirhosTriplets, CartoonStimTrials, SilhouetteTriplets
from plot import plot_similarity_histograms, plot_norm_histogram, plot_similarity_bar
from evaluate import calculate_similarity_totals, shape_bias_rankings
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


def get_embeddings(dir, penult_model, model_type, transform, t, g, s, alpha):
    """ Retrieves embeddings for each image in a dataset from the penultimate
    layer of a given model. Stores the embeddings in a dictionary (indexed by
    image name, eg. cat4-truck3). Returns the dictionary and stores it in a json
    file (model_type_embeddings.json)

    :param dir: path of the dataset
    :param model_type: the type of model, eg. saycam, resnet50, etc.
    :param transform: appropriate transforms for the given model (should match training
                      data stats)
    :param t: true if running triplet simulations
    :param g: true if using grayscale Geirhos dataset
    :param s: true if using silhouette version of Geirhos style transfer
    :param alpha: controls transparency for silhouette stimuli.

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
        dataset = GeirhosStyleTransferDataset(dir, '', transform)
        if g:
            embedding_dir = 'embeddings/' + model_type + '_gray.json'
        elif s:
            dataset = SilhouetteTriplets(transform, alpha)
            alpha_str = dataset.get_alpha_str()
            embedding_dir = 'embeddings/' + model_type + '_silhouette_' + alpha_str + '.json'
        else:
            embedding_dir = 'embeddings/' + model_type + '_embeddings.json'
    else:
        dataset = CartoonStimTrials(transform)
        embedding_dir = 'embeddings/' + model_type + '_cartoon.json'

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        # Iterate over images
        for idx, batch in enumerate(data_loader):
            im = batch[0]
            name = batch[1][0]

            # Pass image into model
            if model_type == 'clipRN50' or model_type == 'clipViTB32' or model_type == 'clipRN50x4'\
                    or model_type == 'clipRN50x16' or model_type == 'clipViTB16':
                embedding = penult_model.encode_image(im)
                embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize the embedding
            else:
                embedding = penult_model(im).numpy().squeeze()

            embedding_dict[name] = embedding.tolist()

    with open(embedding_dir, 'w') as file:
        json.dump(embedding_dict, file)

    return embedding_dict


def generate_fake_triplets(model_type, model, shape_dir, transform, t, g, c, n=230431):
    """Generates fake embeddings that have the same dimensionality as
     model_type for n triplets, then calculates cosine similarity & dot product
     statistics.

     :param model_type: resnet50, saycam, etc.
     :param n: number of fake triplets to generate. Default is the number of
               triplets the real models see.
     :param t: true if doing triplet simulations
     :param g: true if using grayscale Geirhos dataset
     :param c: true if using artificial/cartoon dataset
     :param n: number of triplets to generate"""

    s = False
    # Retrieve embedding magnitude statistics from the real model
    try:
        embeddings = json.load(open('embeddings/' + model_type + '_embeddings.json'))
    except FileNotFoundError:
        embeddings = get_embeddings(shape_dir, model, model_type, transform, t, g, s, alpha)

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
    calculate_similarity_totals(model_type, c, s, 1)


def triplets(model_type, transform, embeddings, verbose, g, s, alpha, shape_dir):
    """First generates all possible triplets of the following form:
    (anchor image, shape match, texture match). Then retrieves the activations
    of the penultimate layer of a given model for each image in the triplet.
    Finally, computes either cosine similarity, dot products, or Euclidean distances:
    anchor x shape match, anchor x texture match. This determines whether the model
    thinks the shape or texture match for an anchor image is closer to the anchor and
    essentially provides a secondary measure of shape/texture bias. This function
    returns a dictionary where values are a list: position 0 contains a dataframe
    of results for a given anchor, and position 1 contains an appropriate path for
    a corresponding CSV file. The keys are the names of the anchor stimuli.

    :param model_type: resnet50, saycam, etc.
    :param transform: appropriate transforms for the given model (should match training
                      data stats)
    :param embeddings: a dictionary of embeddings for each image for the given model
    :param verbose: true if results should be printed to the terminal.
    :param g: true if a grayscale version of the Geirhos dataset should be used.
    :param s: true if the silhouette version of the Geirhos dataset should be used.
    :param alpha: controls transparency for silhouette stimuli.
    :param shape_dir: directory for the Geirhos dataset.

    :return: a dictionary containing a dataframe of results and a path for a CSV file for
             each anchor stimulus.
    """

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

    # Set same_instance=False if you want matches to exclude same instances.
    same_instance = True
    if s:
        t = SilhouetteTriplets(transform, alpha)
        alpha_str = t.get_alpha_str()
    else:
        t = GeirhosTriplets(transform, same_instance=same_instance)  # Default transforms.

    images = t.shape_classes.keys()
    all_triplets = t.triplets_by_image
    results = {key: None for key in images}  # a dictionary of anchor name to dataframe mappings

    metrics = ['dot', 'cos', 'ed']

    columns = ['Model', 'Anchor', 'Anchor Shape', 'Anchor Texture', 'Shape Match', 'Texture Match',
               'Metric', 'Shape Distance', 'Texture Distance', 'Shape Match Closer',
               'Texture Match Closer']

    cosx = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    for anchor in images:  # Iterate over possible anchor images
        anchor_triplets = all_triplets[anchor]['triplets']
        num_triplets = len(anchor_triplets)

        df = pd.DataFrame(index=range(num_triplets * len(metrics)), columns=columns)
        df['Anchor'] = anchor[:-4]
        df['Model'] = model_type
        df['Anchor Shape'] = t.shape_classes[anchor]['shape_spec']
        df['Anchor Texture'] = t.shape_classes[anchor]['texture_spec']

        metric_mult = 0  # Ensures correct placement of results

        for metric in metrics:  # Iterate over distance metrics
            step = metric_mult * num_triplets

            for i in range(num_triplets):  # Iterate over possible triplets
                df.at[i + step, 'Metric'] = metric

                triplet = anchor_triplets[i]
                shape_match = triplet[1]
                texture_match = triplet[2]

                df.at[i + step, 'Shape Match'] = shape_match[:-4]
                df.at[i + step, 'Texture Match'] = texture_match[:-4]

                # Get image embeddings
                anchor_output = torch.FloatTensor(embeddings[anchor])
                shape_output = torch.FloatTensor(embeddings[shape_match])
                texture_output = torch.FloatTensor(embeddings[texture_match])

                if anchor_output.shape[0] == 1:
                    anchor_output = torch.squeeze(anchor_output, 0)
                    shape_output = torch.squeeze(shape_output, 0)
                    texture_output = torch.squeeze(texture_output, 0)

                if metric == 'cos':  # Cosine similarity
                    shape_dist = cosx(anchor_output, shape_output).item()
                    texture_dist = cosx(anchor_output, texture_output).item()
                elif metric == 'dot':  # Dot product
                    shape_dist = np.dot(anchor_output, shape_output).item()
                    texture_dist = np.dot(anchor_output, texture_output).item()
                else:  # Euclidean distance
                    shape_dist = torch.cdist(torch.unsqueeze(shape_output, 0), torch.unsqueeze(anchor_output, 0)).item()
                    texture_dist = torch.cdist(torch.unsqueeze(texture_output, 0),
                                               torch.unsqueeze(anchor_output, 0)).item()

                df.at[i + step, 'Shape Distance'] = shape_dist
                df.at[i + step, 'Texture Distance'] = texture_dist

                if metric == 'ed':
                    shape_dist = -shape_dist
                    texture_dist = -texture_dist

                # Compare shape/texture results
                if shape_dist > texture_dist:
                    df.at[i + step, 'Shape Match Closer'] = 1
                    df.at[i + step, 'Texture Match Closer'] = 0
                else:
                    df.at[i + step, 'Shape Match Closer'] = 0
                    df.at[i + step, 'Texture Match Closer'] = 1

            metric_mult += 1

        if g:
            results[anchor] = [df, 'results/' + model_type + '/grayscale/' + anchor[:-4] + '.csv']
        elif s:
            results[anchor] = [df, 'results/' + model_type + '/silhouette_' + alpha_str + '/' + anchor[:-4] + '.csv']
        elif not same_instance:
            try:
                os.mkdir('results/' + model_type + '/diff_instance')
            except FileExistsError:
                pass
            results[anchor] = [df, 'results/' + model_type + '/diff_instance/' + anchor[:-4] + '.csv']
        else:
            results[anchor] = [df, 'results/' + model_type + '/similarity/' + anchor[:-4] + '.csv']

    return results


def cartoon_stimuli(model_type, transform, embeddings, verbose):
    """
    Runs simulations with cartoon dataset quadruplets. The quadruplets consist of an
    anchor image, a shape match, a color match, and a texture match. Similarity is
    computed between the anchor image and all three matches, and the match with the
    highest similarity is considered as a model "choice." Matches are exclusive; eg.
    the color match does not match the anchor in shape or texture. This function
    returns a dictionary where values are a list: position 0 contains a dataframe
    of results for a given anchor, and position 1 contains an appropriate path for
    a corresponding CSV file. The keys are the names of the anchor stimuli.

    :param model_type: resnet50, saycam, etc.
    :param transform: appropriate transforms for the given model (should match training
                      data stats)
    :param embeddings: a dictionary of embeddings for each image for the given model
    :param verbose: true if results should be printed to the terminal.

    :return: a dictionary containing a dataframe of results and a path for a CSV file for
             each anchor stimulus.
    """

    trials = CartoonStimTrials(transform)
    stimuli = trials.all_stims.keys()
    all_trials = trials.trials_by_image
    results = {key: None for key in stimuli}  # a dictionary of anchor name to dataframe mappings

    metrics = ['dot', 'cos', 'ed']

    columns = ['Model', 'Anchor', 'Anchor Shape', 'Anchor Texture', 'Anchor Color',
               'Shape Match', 'Texture Match', 'Color Match', 'Metric',
               'Shape Distance', 'Texture Distance', 'Color Distance',
               'Shape Match Closer', 'Texture Match Closer', 'Color Match Closer']

    cosx = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    for anchor in stimuli:  # Iterate over all possible anchors
        anchor_trials = all_trials[anchor]['trials']
        num_trials = len(anchor_trials)

        if num_trials == 0:
            continue

        df = pd.DataFrame(index=range(num_trials * len(metrics)), columns=columns)
        df['Anchor'] = anchor
        df['Model'] = model_type
        df['Anchor Shape'] = trials.all_stims[anchor]['shape']
        df['Anchor Texture'] = trials.all_stims[anchor]['texture']
        df['Anchor Color'] = trials.all_stims[anchor]['color']

        metric_mult = 0  # Ensures correct placement of results

        for metric in metrics:  # Iterate over distance metrics
            step = metric_mult * num_trials

            for i in range(num_trials):  # Iterate over possible trials
                df.at[i + step, 'Metric'] = metric

                trial = anchor_trials[i]
                shape_match = trial[1]
                texture_match = trial[2]
                color_match = trial[3]

                df.at[i + step, 'Shape Match'] = shape_match[:-4]
                df.at[i + step, 'Texture Match'] = texture_match[:-4]
                df.at[i + step, 'Color Match'] = color_match[:-4]

                # Get image embeddings
                anchor_output = torch.FloatTensor(embeddings[anchor])
                shape_output = torch.FloatTensor(embeddings[shape_match])
                texture_output = torch.FloatTensor(embeddings[texture_match])
                color_output = torch.FloatTensor(embeddings[color_match])

                if model_type == 'clipRN50' or model_type == 'clipViTB32' or model_type == 'clipRN50x4'\
                        or model_type == 'clipRN50x16' or model_type == 'clipViTB16':
                    anchor_output = torch.squeeze(anchor_output, 0)
                    shape_output = torch.squeeze(shape_output, 0)
                    texture_output = torch.squeeze(texture_output, 0)
                    color_output = torch.squeeze(color_output, 0)

                if metric == 'cos':  # Cosine similarity
                    shape_dist = cosx(anchor_output, shape_output).item()
                    texture_dist = cosx(anchor_output, texture_output).item()
                    color_dist = cosx(anchor_output, color_output).item()
                elif metric == 'dot':  # Dot
                    shape_dist = np.dot(anchor_output, shape_output).item()
                    texture_dist = np.dot(anchor_output, texture_output).item()
                    color_dist = np.dot(anchor_output, color_output).item()
                else:  # Euclidean distance
                    shape_dist = torch.cdist(torch.unsqueeze(shape_output, 0), torch.unsqueeze(anchor_output, 0)).item()
                    texture_dist = torch.cdist(torch.unsqueeze(texture_output, 0), torch.unsqueeze(anchor_output, 0)).item()
                    color_dist = torch.cdist(torch.unsqueeze(color_output, 0), torch.unsqueeze(anchor_output, 0)).item()

                df.at[i + step, 'Shape Distance'] = shape_dist
                df.at[i + step, 'Texture Distance'] = texture_dist
                df.at[i + step, 'Color Distance'] = color_dist

                if metric == 'ed':
                    shape_dist = -shape_dist
                    texture_dist = -texture_dist
                    color_dist = -color_dist

                if shape_dist > texture_dist and shape_dist > color_dist:
                    df.at[i + step, 'Shape Match Closer'] = 1
                    df.at[i + step, 'Texture Match Closer'] = 0
                    df.at[i + step, 'Color Match Closer'] = 0
                elif texture_dist > shape_dist and texture_dist > color_dist:
                    df.at[i + step, 'Shape Match Closer'] = 0
                    df.at[i + step, 'Texture Match Closer'] = 1
                    df.at[i + step, 'Color Match Closer'] = 0
                else:
                    df.at[i + step, 'Shape Match Closer'] = 0
                    df.at[i + step, 'Texture Match Closer'] = 0
                    df.at[i + step, 'Color Match Closer'] = 1

            metric_mult += 1

        results[anchor] = [df, 'results/' + model_type + '/cartoon/' + anchor[:-4] + '.csv']

    return results


def initialize_model(model_type):
    """Initializes the model and puts it into evaluation mode. Returns the model.
    Additionally strips the final layer from the model and returns this as penult_model.
    Finally, retrieves the correct transforms for each model and returns them.

    :param model_type: resnet50, saycam, etc.

    :return: the loaded model in evaluation mode, the model with the penultimate
             layer removed, and the correct transforms for the model (using statistics
             calculated from the specific model's training data)."""

    # These are the ImageNet transforms; most models will use these, but a few redefine them
    transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
        model, transform = clip.load('RN50', device='cpu')
    elif model_type == 'clipRN50x4':
        model, transform = clip.load('RN50x4', device='cpu')
    elif model_type == 'clipRN50x16':
        model, transform = clip.load('RN50x16', device='cpu')
    elif model_type == 'clipViTB32':
        model, transform = clip.load('ViT-B/32', device='cpu')
    elif model_type == 'clipViTB16':
        model, transform = clip.load('ViT-B/16', device='cpu')
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

    # Remove the final layer from the model
    if model_type == 'saycam' or model_type == 'saycamA' or model_type == 'saycamS'\
            or model_type == 'saycamY':
        modules = list(model.module.children())[:-1]
        penult_model = nn.Sequential(*modules)
    elif model_type == 'resnet50' or model_type == 'mocov2' or model_type == 'swav':
        modules = list(model.children())[:-1]
        penult_model = nn.Sequential(*modules)
    elif model_type == 'clipRN50' or model_type == 'clipViTB32' or model_type == 'dino_resnet50'\
            or model_type == 'clipRN50x4' or model_type == 'clipRN50x16' or model_type == 'clipViTB16':
        penult_model = model
    elif model_type == 'alexnet' or model_type == 'vgg16':
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        penult_model = model

    return model, penult_model, transform


def run_simulations(args, model_type):
    """By default: passes images one at a time through a given model and stores/plots the results
    (the shape/texture of the image, the classification made, and whether or not
    the classifcation was a shape classification, a texture classification, or neither.)

    If t flag: runs triplet simulations using the Geirhos dataset
               (see documentation for the triplets function).
    If c flag: runs quadruplet simulations using Brenden Lake's artificial stimulus dataset
               (see documentation for the cartoon_stimuli function).
    If g flag: runs simulations using a grayscale version of the Geirhos dataset.
    If s flag: runs simulations using versions of the Geirhos style transfer dataset with
               white backgrounds.

    By default, the model is the SAYCAM-trained resnext model. This can be changed when running
    this program in the terminal by using -m 'model_type' or the --all flag, which will run
    desired simulations for all available models.

    :param args: command line arguments
    :param model_type: the type of model, saycam by default. Try -m 'resnet50' to change,
        for example."""

    plot = args.plot
    verbose = args.verbose
    t = args.triplets
    c = args.cartoon
    g = args.grayscale
    s = args.silhouettes
    alpha = args.alpha

    if g:
        shape_dir = 'stimuli-shape/style-transfer-gray'
    elif s:
        shape_dir = 'stimuli-shape/texture-silhouettes-' + str(alpha)
    else:
        shape_dir = 'stimuli-shape/style-transfer'

    # Initialize the model and put in evaluation mode; retrieve transforms
    model, penult_model, transform = initialize_model(model_type)

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
                embeddings = get_embeddings(shape_dir, penult_model, model_type, transform, t, g, s, alpha)
        elif s:
            try:
                os.mkdir('results/' + model_type + '/silhouette_' + str(alpha))
            except FileExistsError:
                pass

            try:
                embeddings = json.load(open('embeddings/' + model_type + '_silhouette_' + str(alpha) + '.json'))
            except FileNotFoundError:
                embeddings = get_embeddings(shape_dir, penult_model, model_type, transform, t, g, s, alpha)
        else:
            try:
                os.mkdir('results/' + model_type + '/similarity')
            except FileExistsError:
                pass

            try:
                embeddings = json.load(open('embeddings/' + model_type + '_embeddings.json'))
            except FileNotFoundError:
                embeddings = get_embeddings(shape_dir, penult_model, model_type, transform, t, g, s, alpha)

        results = triplets(model_type, transform, embeddings, verbose, g, s, alpha, shape_dir)

        # Convert result DataFrames to CSV files
        for anchor in results.keys():
            anchor_results = results[anchor]
            df = anchor_results[0]
            path = anchor_results[1]

            df.to_csv(path, index=False)

        if plot:
            plot_similarity_histograms(model_type, g, s)
            plot_norm_histogram(model_type, c, g, s)

    elif c:
        try:
            os.mkdir('results/' + model_type + '/cartoon')
        except FileExistsError:
            pass

        try:
            os.mkdir('figures/' + model_type + '/cartoon')
        except FileExistsError:
            pass

        try:
            embeddings = json.load(open('embeddings/' + model_type + '_cartoon.json'))
        except FileNotFoundError:
            embeddings = get_embeddings('', model, model_type, transform, t, g, s, alpha)

        results = cartoon_stimuli(model_type, transform, embeddings, verbose)

        # Convert result DataFrames to CSV files
        for anchor in results.keys():
            anchor_results = results[anchor]
            if anchor_results is None:
                continue
            df = anchor_results[0]
            path = anchor_results[1]

            df.to_csv(path, index=False)
    '''
    else:  # The code below considers model decisions and not similarities; it isn't used
        shape_dict = dict.fromkeys(shape_categories)  # for storing the results
        shape_categories0 = [shape + '0' for shape in shape_categories]
        shape_dict0 = dict.fromkeys(shape_categories0)

        shape_spec_dict = dict.fromkeys(shape_categories)  # contains lists of specific textures for each shape
        for shape in shape_categories:
            shape_dict[shape] = shape_dict0.copy()
            shape_spec_dict[shape] = []

        # Load and process the images using my custom Geirhos style transfer dataset class
        style_transfer_dataset = GeirhosStyleTransferDataset(shape_dir, texture_dir, transform)
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
    '''


if __name__ == '__main__':
    """This file is used to load models, retrieve image embeddings, and run simulations.
    See the documentation for each function above for more information."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Example: saycam, resnet50', required=False, default='saycam')
    parser.add_argument('-v', '--verbose', help='Prints results.', required=False, action='store_true')
    parser.add_argument('-p', '--plot', help='Plots results.', required=False, action='store_true')
    parser.add_argument('-t', '--triplets', help='Obtains similarities for triplets of images.',
                        required=False, action='store_true')
    parser.add_argument('-s', '--silhouettes', help='Obtains similarities for triplets of silhouette images.',
                        required=False, action='store_true')
    parser.add_argument('--alpha', help='Transparency value for silhouette triplets. 1=no background texture info.'
                                        '0=original Geirhos stimuli.', default=1)
    parser.add_argument('-c', '--cartoon', help='Obtains similarities for cartoon, novel stimuli.',
                        required=False, action='store_true')
    parser.add_argument('-g', '--grayscale', help='Runs simulations with a grayscale version of the Geirhos dataset.',
                        required=False, action='store_true')
    parser.add_argument('--all', help='Generates plots, summaries, or results for all models.', required=False, action='store_true')
    args = parser.parse_args()

    a = args.all
    plot = args.plot
    t = args.triplets
    c = args.cartoon
    s = args.silhouettes
    alpha = args.alpha

    model_list = ['saycam', 'saycamA', 'saycamS', 'saycamY', 'resnet50', 'clipRN50', 'clipRN50x4',
                  'clipRN50x16', 'clipViTB32', 'clipViTB16', 'dino_resnet50', 'alexnet', 'vgg16',
                  'swav', 'mocov2']

    try:
        os.mkdir('results')
    except FileExistsError:
        pass

    try:
        os.mkdir('figures')
    except FileExistsError:
        pass

    if a:
        if not plot:
            for model_type in model_list:
                print("Running simulations for {0}".format(model_type))
                run_simulations(args, model_type)
                calculate_similarity_totals(model_type, c, s, alpha)

            print("\nCalculating ranks...")
            if t:
                shape_bias_rankings('similarity')
            elif s:
                shape_bias_rankings('silhouette', alpha=alpha)
            elif c:
                shape_bias_rankings('cartoon')

        else:
            g = args.grayscale
            plot_similarity_bar(g, c, s, alpha)

    else:
        run_simulations(args, args.model)
        calculate_similarity_totals(args.model, c, s, alpha)
