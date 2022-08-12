import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel, logging, ViTConfig
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np
import pandas as pd
from data import SilhouetteTriplets
#from plot import plot_similarity_bar, plot_bias_charts
from evaluate import calculate_similarity_totals, csv_class_values, calculate_totals, calculate_proportions, get_num_draws
import clip
import probabilities_to_decision
from random import sample
logging.set_verbosity_error()  # Suppress warnings from Hugging Face


# Default model list. Note that if new models are added, initialization
# code for the new models will need to be added to the initialize_model
# and get_embeddings functions. Furthermore, styles will need to be
# added to the plotting functions (eg. marker shape, line color, etc.)
model_list = ['saycam', 'resnet50', 'clipViTB16', 'dino_resnet50',
              'ViTB16', 'resnet50_random', 'ViTB16_random']


class LinearClassifier(nn.Module):
    """Linear layer to place on top of dino_resnet50."""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def get_model_list():
    return model_list


def new_seed(args, stimuli_dir):
    """ This function generates num_draws (see evaluate.py) random selections
    of triplets and stores them. The purpose of this is to ensure that all
    models are seeing the same random draw of triplets until this function
    is called again. Shape/texture similarity proportions for all models are
    calculated as the averages of shape/texture similarity proportions of a
    number of random draws of triplets.

    :param args: command line arguments
    :param stimuli_dir: location of relevant dataset"""

    if args.novel:
        num_draws = 1
        seed_path = 'novel_seed.json'
    else:
        num_draws = get_num_draws()
        seed_path = 'seed.json'

    selections = {i: None for i in range(num_draws)}
    d = SilhouetteTriplets(args, stimuli_dir, None)

    cap = d.max_num_triplets()

    for i in range(num_draws):
        selection = {}

        for anchor in d.triplets_by_image.keys():
            all_triplets = d.triplets_by_image[anchor]['triplets']
            selection[anchor] = sample(all_triplets, cap)

        selections[i] = selection

    with open(seed_path, 'w') as file:
        json.dump(selections, file)


def get_embeddings(args, stimuli_dir, model_type, penult_model, transform, n=-1):
    """ Retrieves embeddings for each image in a dataset from the penultimate
    layer of a given model. Stores the embeddings in a dictionary (indexed by
    image name, eg. cat4-truck3). Returns the dictionary and stores it in a json
    file (model_type_embeddings.json)

    :param args: command line arguments
    :param stimuli_dir: path of the dataset
    :param model_type: the type of model, eg. saycam, resnet50, etc.
    :param penult_model: the model with the last layer removed.
    :param transform: appropriate transforms for the given model (should match training
        data stats)
    :param n: for use when model_type = resnet50_random or ViTB16_random. Specifies
              which particular random model to use.

    :return: a dictionary indexed by image name that contains the embeddings for
        all images in a dataset extracted from the penultimate layer of a given
        model.
    """

    if 'random' in model_type:
        model_type = '{0}_{1}'.format(model_type, n)

    bg = args.bg

    try:
        os.mkdir('embeddings')
    except FileExistsError:
        pass

    try:
        os.mkdir('embeddings/{0}'.format(model_type))
    except FileExistsError:
        pass

    if bg:
        bg_str = 'background_{0}'.format(bg)

        try:
            os.mkdir('embeddings/{0}/{1}'.format(model_type, bg_str))
        except FileExistsError:
            pass

    embedding_dir = 'embeddings/{0}/{1}.json'.format(model_type, stimuli_dir)

    try:
        embeddings = json.load(open(embedding_dir))
        return embeddings

    except FileNotFoundError:  # Retrieve and store embeddings
        # Initialize dictionary
        embeddings = {}

        dataset = SilhouetteTriplets(args, stimuli_dir, transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            # Iterate over images
            for idx, batch in enumerate(data_loader):
                im = batch[0]
                name = batch[1][0]

                # Pass image into model
                if model_type == 'clipViTB16':
                    embedding = penult_model.encode_image(im)
                    embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize the embedding
                elif model_type == 'ViTB16' or 'ViTB16_random' in model_type:
                    im['pixel_values'] = im['pixel_values'].squeeze(0)
                    outputs = penult_model(**im)
                    embedding = outputs.last_hidden_state.squeeze(0)[0, :]
                else:
                    embedding = penult_model(im).numpy().squeeze()

                embeddings[name] = embedding.tolist()

        with open(embedding_dir, 'w') as file:
            json.dump(embeddings, file)

        return embeddings


def triplets(args, model_type, stimuli_dir, embeddings, n=-1):
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

    :param args: command line arguments
    :param model_type: resnet50, saycam, etc.
    :param stimuli_dir: location of dataset
    :param embeddings: a dictionary of embeddings for each image for the given model
    :param n: for use when model_type = resnet50_random or ViTB16_random. Specifies
              which particular random model to use.

    :return: a dictionary containing a dataframe of results and a path for a CSV file for
             each anchor stimulus.
    """

    dataset = SilhouetteTriplets(args, stimuli_dir, None)

    images = dataset.shape_classes.keys()
    all_triplets = dataset.triplets_by_image
    results = {key: None for key in images}  # a dictionary of anchor name to dataframe mappings

    metrics = ['dot', 'cos', 'ed']

    columns = ['Model', 'Anchor', 'Anchor Shape', 'Anchor Texture', 'Shape Match', 'Texture Match',
               'Metric', 'Shape Distance', 'Texture Distance', 'Shape Match Closer',
               'Texture Match Closer']

    cosx = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    if 'random' in model_type:
        model_type = '{0}_{1}'.format(model_type, n)

    for anchor in images:  # Iterate over possible anchor images
        anchor_triplets = all_triplets[anchor]['triplets']
        num_triplets = len(anchor_triplets)

        df = pd.DataFrame(index=range(num_triplets * len(metrics)), columns=columns)
        df['Anchor'] = anchor[:-4]
        df['Model'] = model_type
        if args.novel:
            df['Anchor Shape'] = dataset.shape_classes[anchor]['shape']
            df['Anchor Texture'] = dataset.shape_classes[anchor]['texture']
        else:
            df['Anchor Shape'] = dataset.shape_classes[anchor]['shape_spec']
            df['Anchor Texture'] = dataset.shape_classes[anchor]['texture_spec']

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

        results[anchor] = [df, 'results/{0}/{1}/{2}.csv'.format(model_type, stimuli_dir, anchor[:-4])]

    return results


def initialize_model(model_type, n=-1):
    """Initializes the model and puts it into evaluation mode. Returns the model.
    Additionally strips the final layer from the model and returns this as penult_model.
    Finally, retrieves the correct transforms for each model and returns them.

    :param model_type: resnet50, saycam, etc.
    :param n: for use when model_type = resnet50_random or ViTB16_random. Specifies
              which particular random model to use.

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
        # Load Emin Ohran's pretrained SAYCAM model + ImageNet classifier from its .tar file
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        model = nn.DataParallel(model)
        checkpoint = torch.load('models/fz_IN_resnext50_32x4d_augmentation_True_SAY_5_288.tar',
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_type == 'resnet50_random':
        model = models.resnet50(pretrained=False)

        try:
            model.load_state_dict(torch.load('models/resnet50_random_{0}.pth'.format(n)))
        except FileNotFoundError:
            torch.save(model.state_dict(), 'models/resnet50_random_{0}.pth'.format(n))
    elif model_type == 'clipViTB16':
        model, transform = clip.load('ViT-B/16', device='cpu')
    elif model_type == 'dino_resnet50':
        model = models.resnet50(pretrained=False)
        checkpoint = torch.load('models/dino_resnet50_linearweights.pth',
                                map_location=torch.device('cpu'))['state_dict']
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        linear_classifier = LinearClassifier(model.fc.weight.shape[1], num_labels=1000)
        linear_classifier.load_state_dict(checkpoint)
        model = linear_classifier
        penult_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif model_type == 'ViTB16':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        penult_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # Note: "transform" for the ViT model is not actually a transform
        transform = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    elif model_type == 'ViTB16_random':
        configuration = ViTConfig()
        model = ViTForImageClassification(configuration)

        try:
            model.load_state_dict(torch.load('models/ViTB16_random_{0}.pth'.format(n)))
        except FileNotFoundError:
            torch.save(model.state_dict(), 'models/ViTB16_random_{0}.pth'.format(n))

        penult_model = ViTModel(configuration)
        transform = ViTFeatureExtractor()
    else:
        print('The model ' + model_type + ' has not yet been defined. Please see main.py')
        sys.exit()

    # Put model in evaluation mode
    model.eval()

    # Remove the final layer from the model
    if model_type == 'saycam':
        modules = list(model.module.children())[:-1]
        penult_model = nn.Sequential(*modules)
    elif model_type == 'resnet50' or model_type == 'resnet50_random':
        modules = list(model.children())[:-1]
        penult_model = nn.Sequential(*modules)
    elif model_type == 'clipViTB16':
        penult_model = model

    return model, penult_model, transform


def clip_predictions(im, model, model_type):
    """Gives probabilities for ImageNet classes for a CLIP model given an
    input image.

    :param im: the image to obtain probabilities for
    :param model: the CLIP model to obtain probabilties from
    :param model_type: 'clipRN50', 'clipRN50x4', 'clipRN50x16', 'clipViTB32',
                       or 'clipViTB16'

    :return: a 1x1000 dim tensor of probabilities for ImageNet classes"""

    try:
        text_features = torch.load('embeddings/{0}_text_features.pt'.format(model_type))
    except FileNotFoundError:
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        text = clip.tokenize(categories)
        text_features = model.encode_text(text)
        torch.save(text_features, 'embeddings/{0}_text_features.pt'.format(model_type))

    image_features = model.encode_image(im)

    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100 * image_features @ text_features.T)

    return similarity


def run_simulations(args, model_type, stimuli_dir, n=-1):
    """By default: passes images one at a time through a given model and stores/plots the results
    (the shape/texture of the image, the classification made, and whether or not
    the classifcation was a shape classification, a texture classification, or neither.)

    If not classification flag: runs triplet simulations using the Geirhos dataset
               (see documentation for the triplets function).
        - If novel flag, novel shape stimuli will be used.
        - If bg flag, stimuli over a background image will be used.

    By default, the model is the SAYCAM-trained resnext model. This can be changed when running
    this program in the terminal by using -m 'model_type' or the --all flag, which will run
    desired simulations for all available models.

    :param args: command line arguments
    :param model_type: the type of model, saycam by default. Try -m 'resnet50' to change,
        for example.
    :param stimuli_dir: location of the stimuli for this simulation.
    :param n: for use when model_type = resnet50_random or ViTB16_random. Specifies
              which particular random model to use."""

    #plot = args.plot
    classification = args.classification
    bg = args.bg

    clip_list = ['clipViTB16']

    # Initialize the model and put in evaluation mode; retrieve transforms
    model, penult_model, transform = initialize_model(model_type, n=n)

    # Create directories for results and plots
    try:
        os.mkdir('results/{0}'.format(model_type))
    except FileExistsError:
        pass

    if classification:
        class_str = 'classifications/'
        try:
            os.mkdir('results/{0}/classifications'.format(model_type))
        except FileExistsError:
            pass
    else:
        class_str = ''

    if bg:
        bg_str = '{0}background_{1}'.format(class_str, bg)

        try:
            os.mkdir('results/{0}/{1}'.format(model_type, bg_str))
        except FileExistsError:
            pass

    try:
        os.mkdir('results/{0}/{1}{2}'.format(model_type, class_str, stimuli_dir))
    except FileExistsError:
        pass

    try:
        os.mkdir('figures/{0}'.format(model_type))
    except FileExistsError:
        pass

    if classification:
        class_str = 'classifications/'
        try:
            os.mkdir('figures/{0}/classifications'.format(model_type))
        except FileExistsError:
            pass
    else:
        class_str = ''

    if bg:
        bg_str = '{0}background_{1}'.format(class_str, bg)

        try:
            os.mkdir('figures/{0}/{1}'.format(model_type, bg_str))
        except FileExistsError:
            pass

    try:
        os.mkdir('figures/{0}/{1}{2}'.format(model_type, class_str, stimuli_dir))
    except FileExistsError:
        pass

    # Run simulations
    if not classification:
        embeddings = get_embeddings(args, stimuli_dir, model_type, penult_model, transform, n=n)

        results = triplets(args, model_type, stimuli_dir, embeddings, n=n)

        # Convert result DataFrames to CSV files
        for anchor in results.keys():
            anchor_results = results[anchor]
            df = anchor_results[0]
            path = anchor_results[1]

            df.to_csv(path, index=False)

        # TODO: PLOTTING CODE INSERTED FOR INDIVIDUAL MODEL PLOTS eg. geirhos style charts

    else:  # Run simulations in the style of Geirhos et al.; ie., obtain classifications
        result_dir = 'results/{0}/{1}{2}'.format(model_type, class_str, stimuli_dir)

        shape_categories = sorted(['knife', 'keyboard', 'elephant', 'bicycle', 'airplane',
                                   'clock', 'oven', 'chair', 'bear', 'boat', 'cat',
                                   'bottle', 'truck', 'car', 'bird', 'dog'])

        shape_dict = dict.fromkeys(shape_categories)  # for storing the results
        shape_categories0 = [shape + '0' for shape in shape_categories]
        shape_dict0 = dict.fromkeys(shape_categories0)

        shape_spec_dict = dict.fromkeys(shape_categories)  # contains lists of specific textures for each shape
        for shape in shape_categories:
            shape_dict[shape] = shape_dict0.copy()
            shape_spec_dict[shape] = []

        dataset = SilhouetteTriplets(args, stimuli_dir, transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Obtain ImageNet - Geirhos mapping
        mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
        softmax = nn.Softmax(dim=1)
        softmax2 = nn.Softmax(dim=0)

        with torch.no_grad():
            # Pass images into the model one at a time
            for batch in dataloader:
                im, name = batch
                split_name = name[0].split('-')

                shape = ''.join([i for i in split_name[0] if not i.isdigit()])
                texture = ''.join([i for i in split_name[1][:-4] if not i.isdigit()])
                texture_spec = split_name[1][:-4]

                if model_type == 'ViTB16':
                    output = model(**im)
                    output = output.logits
                elif model_type in clip_list:
                    output = clip_predictions(im, model, model_type)
                elif model_type == 'dino_resnet50' or model_type == 'swav':
                    embed = penult_model(im)
                    output = model(embed)
                else:
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

                shape_dict[shape][texture_spec + '0'] = [decision, class_values,
                                                         decision_restricted, restricted_class_values]
                shape_spec_dict[shape].append(texture_spec)

            csv_class_values(shape_dict, shape_categories, shape_spec_dict, result_dir)
            calculate_totals(shape_categories, result_dir)
            calculate_proportions(model_type, result_dir)


if __name__ == '__main__':
    """This file is used to load models, retrieve image embeddings, and run simulations.
    See the documentation for each function above for more information."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Example: saycam, resnet50', required=False, default='saycam')
    parser.add_argument('-p', '--plot', help='Plots results.', required=False, action='store_true')
    parser.add_argument('--classification', help='Obtains classification decisions. Otherwise, obtains similarities '
                                                 'for triplets of images (default).', required=False,
                        action='store_true')
    parser.add_argument('--percent_size', help='Controls the size of the stimuli.', required=False, default='100')
    parser.add_argument('--unaligned', help='Randomly place the stimuli. Otherwise, stimuli are placed'
                                            'in the center of the image.', required=False, action='store_true')
    parser.add_argument('--novel', help='Uses novel shape/texture stimuli triplets. This flag must be used with '
                                        'the -s flag.', required=False, action='store_true')
    parser.add_argument('--bg', help='Runs silhouette triplet simulations (-s, -t) using stimuli with an image '
                                     'background.', required=False, default=None)
    parser.add_argument('--alpha', help='Transparency value for silhouette triplets. 1=no background texture info.'
                                        '0=original Geirhos stimuli.', default=1, type=float)
    parser.add_argument('--all_models', help='Generates plots, summaries, or results for all models.', required=False,
                        action='store_true')
    parser.add_argument('--new_seed', help='Generates a new collection of randomly selected triplets to use in the '
                                           'calculation of similarity shape/texture bias proportions.', required=False,
                        action='store_true')
    args = parser.parse_args()

    model = args.model
    plot = args.plot
    classification = args.classification
    percent = args.percent_size
    unaligned = args.unaligned
    novel = args.novel
    bg = args.bg
    alpha = args.alpha
    all_models = args.all_models
    new_seed = args.new_seed

    N = 10  # number of random models to test and average over

    # Do not classify novel stimuli
    assert not (novel and classification)

    # Size/alignment/background should only be varied for alpha=1 stimuli.
    if alpha != 1:
        percent_size = '100'
        unaligned = False
        bg = None

    try:
        os.mkdir('results')
    except FileExistsError:
        pass

    try:
        os.mkdir('figures')
    except FileExistsError:
        pass

    try:
        os.mkdir('shape_classes')
    except FileExistsError:
        pass

    if bg:
        bg_str = 'background_{0}/'.format(bg)

        try:
            os.mkdir('stimuli/{0}'.format(bg_str))
        except FileExistsError:
            pass

        try:
            os.mkdir('shape_classes/{0}'.format(bg_str))
        except FileExistsError:
            pass
    else:
        bg_str = ''

    if novel:
        if unaligned:
            stimuli_dir = '{0}novel-alpha{1}-size{2}-unaligned'.format(bg_str, alpha, percent)
        else:
            stimuli_dir = '{0}novel-alpha{1}-size{2}-aligned'.format(bg_str, alpha, percent)
    else:
        if unaligned:
            stimuli_dir = '{0}geirhos-alpha{1}-size{2}-unaligned'.format(bg_str, alpha, percent)
        else:
            stimuli_dir = '{0}geirhos-alpha{1}-size{2}-aligned'.format(bg_str, alpha, percent)

    if new_seed or not os.path.exists('seed.json') or not os.path.exists('novel_seed.json'):
        new_seed(args, stimuli_dir)

    if all_models:
        if not plot:
            for model_type in model_list:
                print("Running simulations for {0}...".format(model_type))

                if 'random' in model_type:
                    for i in range(1, N+1):
                        print('\t{0}_{1}...'.format(model_type, i))
                        run_simulations(args, model_type, stimuli_dir, n=i)
                        calculate_similarity_totals(args, model_type, stimuli_dir, n=i)
                else:
                    run_simulations(args, model_type, stimuli_dir)

                if not classification:
                    calculate_similarity_totals(args, model_type, stimuli_dir, N=N)
        else:
            #plot_similarity_bar(s, alpha, novel=novel, bg=bg)
            #plot_bias_charts(classification)
            print('hi')

    else:
        if 'random' in model:
            for i in range(1, N + 1):
                print('{0}_{1}...'.format(model, i))
                run_simulations(args, model, stimuli_dir, n=i)
                calculate_similarity_totals(args, model, stimuli_dir, n=i)
        else:
            run_simulations(args, model, stimuli_dir)

        if not classification:
            calculate_similarity_totals(args, model, stimuli_dir, N=N)
