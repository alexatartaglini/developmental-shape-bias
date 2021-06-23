import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import PIL
import copy
import os
import json
import numpy as np
import argparse
import pandas as pd
import probabilities_to_decision
import helper.human_categories as sc
import matplotlib.pyplot as plt
import glob
from matplotlib import cm
import random
from scipy import spatial
from data import GeirhosStyleTransferDataset, GeirhosTriplets


def plot_class_values(categories, class_values, im, shape, texture, model_type):
    """This function plots the values that the model assigns to the Geirhos
    Style Transfer classes (airplane, bear, ..., oven, truck; 16 total).

    :param categories: a list of the 16 Geirhos classes, either organized by shape or
        texture.
    :param class_values: A length 16 vector. The values referred to here are those
        calculated by the Geirhos probability mapping code, which takes the 1000-length
        vector output of the model as an input; it then groups the various ImageNet classes
        into groups that correspond with a single Geirhos class, and it takes the average of
        the probabilities amongst this group of ImageNet classes. This average becomes
        the value assigned to the Geirhos class, and the class receiving the highest average
        probability is the model's decision.
    :param im: the path of the image file that produced these results.
    :param shape: the shape classification of the given image.
    :param texture: the texture classification of the given image."""

    decision_idx = class_values.index(max(class_values))  # index of maximum class value
    decision = categories[decision_idx]
    shape_idx = categories.index(shape)  # index of shape category
    texture_idx = categories.index(texture)  # index of texture category

    spec = plt.GridSpec(ncols=2, nrows=1, width_ratios=[4, 1], wspace=0.2, )

    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(9.5)

    # Bar plot
    fig.add_subplot(spec[0])
    plt.bar(categories, class_values, color=(0.4, 0.4, 0.4), width=0.4)
    plt.bar(categories[decision_idx], class_values[decision_idx],
            color=(0.9411764705882353, 0.00784313725490196, 0.4980392156862745), width=0.4)
    plt.bar(categories[shape_idx], class_values[shape_idx],
            color=(0.4980392156862745, 0.788235294117647, 0.4980392156862745), width=0.4)
    plt.bar(categories[texture_idx], class_values[texture_idx],
            color=(0.7450980392156863, 0.6823529411764706, 0.8313725490196079), width=0.4)
    plt.xlabel("Geirhos Style Transfer class", fontsize=12)
    plt.ylabel("Average probabilities across associated ImageNet classes", fontsize=10)
    plt.suptitle("Model decision for " + im + ":", fontsize=15)
    plt.title("Model Outputs", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create the legend
    colors = {'model decision: ' + decision: (0.9411764705882353, 0.00784313725490196, 0.4980392156862745),
              'shape category: ' + shape: (0.4980392156862745, 0.788235294117647, 0.4980392156862745),
              'texture category: ' + texture: (0.7450980392156863, 0.6823529411764706, 0.8313725490196079)}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    # Plot the image
    im_ax = fig.add_subplot(spec[1])
    img = plt.imread('stimuli-shape/style-transfer/' + shape + '/' + im)
    plt.imshow(img)
    plt.title(im)
    im_ax.set_xticks([])
    im_ax.set_yticks([])

    plt.savefig('figures/' + model_type + '/' + im)


def csv_class_values(shape_dict, shape_categories, shape_spec_dict, csv_dir):
    """Writes the shape category, texture category, and model decision for all
    shape-texture combinations in a given Geirhos shape class to a CSV file.
    Also includes whether or not neither the shape or texture classification is made.

    :param shape_dict: a dictionary of values with shape category keys. Should
        store the decision made, the length 16 vector of class values for a
        given shape-image combination, and the decision made when restricted to
        only the shape and texture categories.
    :param shape_categories: a list of all Geirhos shape classes.
    :param shape_spec_dict: a shape-indexed dictionary of lists, each containing
        the specific textures for a given shape (eg. clock1, oven2, instead of
        just clock, oven, etc). This ensures that results for clock1 and clock2
        for example do not overwrite each other.
    :param csv_dir: directory for storing the CSV."""

    columns = ['Shape', 'Texture', 'Decision', 'Shape Category Value', 'Texture Category Value',
               'Decision Category Value', 'Shape Decision', 'Texture Decision', 'Neither',
               'Restricted Decision', 'Restriced Shape Value', 'Restricted Texture Value',
               'Restricted Shape Decision', 'Restricted Texture Decision']

    for shape in shape_categories:
        specific_textures = shape_spec_dict[shape]
        df = pd.DataFrame(index=range(len(specific_textures)), columns=columns)
        df['Shape'] = shape

        for i, row in df.iterrows():
            texture = specific_textures[i]
            decision = shape_dict[shape][texture + '0'][0]
            class_values = shape_dict[shape][texture + '0'][1]
            decision_restricted = shape_dict[shape][texture + '0'][2]
            restricted_class_values = shape_dict[shape][texture + '0'][3]

            row['Texture'] = texture
            row['Decision'] = decision
            row['Shape Category Value'] = class_values[shape_categories.index(shape)]
            row['Texture Category Value'] = class_values[shape_categories.index(texture[:-1:])]
            row['Decision Category Value'] = class_values[shape_categories.index(decision)]

            row['Shape Decision'] = int(decision == shape)
            row['Texture Decision'] = int(decision == texture[:-1:])
            row['Neither'] = int(decision != shape and decision != texture[:-1:])

            row['Restricted Decision'] = decision_restricted
            row['Restricted Shape Decision'] = int(shape == decision_restricted)

            row['Restricted Texture Decision'] = int(texture[:-1:] == decision_restricted)
            row['Restricted Shape Value'] = restricted_class_values[0]
            row['Restricted Texture Value'] = restricted_class_values[1]

        df.to_csv(csv_dir + '/' + shape + '.csv', index=False)


def calculate_totals(shape_categories, result_dir, verbose=False):
    """Calculates the total number of shape, texture, and neither shape nor
    texture decisions by Geirhos shape class (and overall). Stores these
    results in a CSV and optionally prints them out.

    :param shape_categories: a list of Geirhos shape classes.
    :param result_dir: where to store the results.
    :param verbose: True if you want to print the results as well as store them."""

    shape_dict = dict.fromkeys(shape_categories)
    texture_dict = dict.fromkeys(shape_categories)
    neither_dict = dict.fromkeys(shape_categories)
    restricted_shape_dict = dict.fromkeys(shape_categories)
    restricted_texture_dict = dict.fromkeys(shape_categories)

    columns = ['Shape Category', 'Number Shape Decisions', 'Number Texture Decisions',
               'Number Neither', 'Number Restricted Shape Decisions',
               'Number Restricted Texture Decisions', 'Total Number Stimuli']
    result_df = pd.DataFrame(columns=columns, index=range(len(shape_categories) + 1))

    for shape in shape_categories:
        shape_dict[shape] = 0
        texture_dict[shape] = 0
        neither_dict[shape] = 0
        restricted_shape_dict[shape] = 0
        restricted_texture_dict[shape] = 0

    for filename in os.listdir(result_dir):
        if filename[-4:] != '.csv' or filename == 'totals.csv':
            continue

        df = pd.read_csv(result_dir + '/' + filename)
        shape = df['Shape'][0]
        for i, row in df.iterrows():
            if row['Restricted Shape Decision'] != row['Restricted Texture Decision']:
                shape_dict[shape] = shape_dict[shape] + row['Shape Decision']
                texture_dict[shape] += row['Texture Decision']
                neither_dict[shape] += row['Neither']
                restricted_shape_dict[shape] += row['Restricted Shape Decision']
                restricted_texture_dict[shape] += row['Restricted Texture Decision']

    for shape in shape_categories:
        if verbose:
            print("Shape category: " + shape)
            print("\tNumber shape decisions: " + str(shape_dict[shape]))
            print("\tNumber texture decisions: " + str(texture_dict[shape]))
            print("\tNumber neither shape nor texture decisions: " + str(neither_dict[shape]))
            print("\t---------------------------------------------")
            print("\tNumber shape decisions (restricted to only shape/texture classes): "
                  + str(restricted_shape_dict[shape]))
            print("\tNumber texture decisions (restricted to only shape/texture classes): "
                  + str(restricted_texture_dict[shape]))
            print()

        shape_idx = shape_categories.index(shape)
        result_df.at[shape_idx, 'Shape Category'] = shape
        result_df.at[shape_idx, 'Number Shape Decisions'] = shape_dict[shape]
        result_df.at[shape_idx, 'Number Texture Decisions'] = texture_dict[shape]
        result_df.at[shape_idx, 'Number Neither'] = neither_dict[shape]
        result_df.at[shape_idx, 'Number Restricted Shape Decisions'] = restricted_shape_dict[shape]
        result_df.at[shape_idx, 'Number Restricted Texture Decisions'] = restricted_texture_dict[shape]
        result_df.at[shape_idx, 'Total Number Stimuli'] = shape_dict[shape] + texture_dict[shape] +\
                                                          neither_dict[shape]

    if verbose:
        print("IN TOTAL:")
        print("\tNumber shape decisions: " + str(sum(shape_dict.values())))
        print("\tNumber texture decisions: " + str(sum(texture_dict.values())))
        print("\tNumber neither shape nor texture decisions: " + str(sum(neither_dict.values())))
        print("\t---------------------------------------------")
        print("\tNumber shape decisions (restricted to only shape/texture classes): "
              + str(sum(restricted_shape_dict.values())))
        print("\tNumber texture decisions (restricted to only shape/texture classes): "
              + str(sum(restricted_texture_dict.values())))
        print()

    idx = len(shape_categories)  # final row
    result_df.at[idx, 'Shape Category'] = 'total'
    result_df.at[idx, 'Number Shape Decisions'] = sum(shape_dict.values())
    result_df.at[idx, 'Number Texture Decisions'] = sum(texture_dict.values())
    result_df.at[idx, 'Number Neither'] = sum(neither_dict.values())
    result_df.at[idx, 'Total Number Stimuli'] = sum(neither_dict.values()) + \
                                                sum(texture_dict.values()) + sum(shape_dict.values())
    result_df.at[idx, 'Number Restricted Shape Decisions'] = sum(restricted_shape_dict.values())
    result_df.at[idx, 'Number Restricted Texture Decisions'] = sum(restricted_texture_dict.values())

    result_df.to_csv(result_dir + '/totals.csv', index=False)


def calculate_proportions(result_dir, verbose=False):
    """Calculates the proportions of shape and texture decisions for a given model.
    There are two proportions calculated for both shape and texture: 1) with neither
    shape nor texture decisions included, and 2) without considering 'neither'
    decisions. Stores these proportions in a text file and optionally prints them.

    :param result_dir: the directory of the results for the model."""

    df = pd.read_csv(result_dir + '/totals.csv')
    row = df.loc[df['Shape Category'] == 'total']
    shape = int(row['Number Shape Decisions'])
    texture = int(row['Number Texture Decisions'])
    total = int(row['Total Number Stimuli'])

    shape_restricted = int(row['Number Restricted Shape Decisions']) / total
    texture_restricted = int(row['Number Restricted Texture Decisions']) / total

    shape_texture = shape / (shape + texture)
    texture_shape = texture / (shape + texture)
    shape_all = shape / total
    texture_all = texture / total

    strings = ["Proportion of shape decisions (disregarding 'neither' decisions): " + str(shape_texture),
               "Proportion of texture decisions (disregarding 'neither' decisions): " + str(texture_shape),
               "Proportion of shape decisions (including 'neither' decisions): " + str(shape_all),
               "Proportion of texture decisions (including 'neither' decisions): " + str(texture_all),
               "Proportion of shape decisions (restricted to only shape/texture classes): " + str(shape_restricted),
               "Proportion of texture decisions (restricted to only shape/texture classes): " + str(texture_restricted)]
    file = open(result_dir + '/proportions.txt', 'w')

    for i in range(len(strings)):
        file.write(strings[i] + '\n')
        if verbose:
            print(strings[i])

    file.close()


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


def plot_similarity_averages(model_type, shape_categories):
    """Plots average shape/texture dot products and cosine similarities by
    anchor image shape. UNFINISHED

    :param model_type: saycam, resnet50, etc.
    :param shape_categories: a list of the 16 Geirhos classes.
    """

    plot_dir = 'figures/' + model_type + '/similarity'
    try:
        os.mkdir(plot_dir)
    except FileExistsError:
        pass

    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(9.5)


def calculate_similarity_totals(model_type):
    """Calculates proportion of times the shape/texture dot product/cosine similarity
    is closer for a given model. Stores proportions as a csv.

    :param model_type: saycam, resnet50, etc."""

    sim_dir = 'results/' + model_type + '/similarity/'

    shape_dot = 0
    shape_cos = 0
    texture_dot = 0
    texture_cos = 0
    num_rows = 0

    columns = ['Model', 'Shape Dot Closer', 'Shape Cos Closer', 'Texture Dot Closer', 'Texture Cos Closer']
    results = pd.DataFrame(index=range(1), columns=columns)
    results.at[0, 'Model'] = model_type

    for file in glob.glob(sim_dir + '*.csv'):

        if file == sim_dir + 'averages.csv':
            continue
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            shape_dot += int(row['Shape Dot Closer'])
            shape_cos += int(row['Shape Cos Closer'])
            texture_dot += int(row['Texture Dot Closer'])
            texture_cos += int(row['Texture Cos Closer'])
            num_rows += 1

    results.at[0, 'Shape Dot Closer'] = shape_dot / num_rows
    results.at[0, 'Shape Cos Closer'] = shape_cos / num_rows
    results.at[0, 'Texture Dot Closer'] = texture_dot / num_rows
    results.at[0, 'Texture Cos Closer'] = texture_cos / num_rows

    results.to_csv(sim_dir + 'proportions.csv', index=False)


def calculate_similarity_averages(model_type, shape_categories, plot):
    """Calculates average dot product/cosine similarity between an anchor image shape
    class and its shape/texture matches. Stores results in a csv. Optionally generates
    a plot.

    :param model_type: resnet50, saycam, etc.
    :param shape_categories: a list of the 16 Geirhos classes.
    :param plot: true if plot should be generated.
    """

    columns = ['Model', 'Anchor Image Shape', 'Average Dot Shape', 'Average Cos Shape',
               'Average Dot Texture', 'Average Cos Texture']
    results = pd.DataFrame(index=range(len(shape_categories)), columns=columns)
    result_dir = 'results/' + model_type + '/similarity'

    results.at[:, 'Model'] = model_type

    for i in range(len(shape_categories)):  # Iterate over anchor image shapes
        anchor_shape = shape_categories[i]

        shape_dot = 0
        shape_cos = 0
        texture_dot = 0
        texture_cos = 0
        num_triplets = 0

        for file in glob.glob(result_dir + '/' + anchor_shape + '*.csv'):  # Sum results by shape
            df = pd.read_csv(file)

            for index, row in df.iterrows():
                shape_dot += float(row['Shape Dot'])
                shape_cos += float(row['Shape Cos'])
                texture_dot += float(row['Texture Dot'])
                texture_cos += float(row['Texture Cos'])
                num_triplets += 1

        shape_dot = shape_dot / num_triplets
        shape_cos = shape_cos / num_triplets
        texture_dot = texture_dot / num_triplets
        texture_cos = texture_cos / num_triplets

        results.at[i, 'Anchor Image Shape'] = anchor_shape
        results.at[i, 'Average Dot Shape'] = shape_dot
        results.at[i, 'Average Cos Shape'] = shape_cos
        results.at[i, 'Average Dot Texture'] = texture_dot
        results.at[i, 'Average Cos Texture'] = texture_cos

    results.to_csv(result_dir + '/averages.csv', index=False)


def triplets(model_type, embeddings, verbose, shape_dir):
    """First generates all possible triplets of the following form:
    (anchor image, shape match, texture match). Then retrieves the activations
    of the penultimate layer of a given model for each image in the triplet.
    Finally, computes and stores cosine similarity and dot products: anchor x shape match,
    anchor x texture match. This determines whether the model thinks the shape or texture
    match for an anchor image is closer to the anchor and essentially provides a secondary
    measure of shape/texture bias.

    :param model_type: resnet50, saycam, etc.
    :param embeddings: a dictionary of embeddings for each image for the given model
    :param verbose: true if results should be printed to the terminal.
    :param shape_dir: directory for the Geirhos dataset."""

    t = GeirhosTriplets(shape_dir)
    images = t.shape_classes.keys()
    all_triplets = t.triplets_by_image

    sim_dict = {}

    columns = ['Model', 'Anchor', 'Anchor Shape', 'Anchor Texture', 'Shape Match',
               'Texture Match', 'Shape Dot', 'Shape Cos',
               'Texture Dot', 'Texture Cos', 'Shape Dot Closer', 'Shape Cos Closer',
               'Texture Dot Closer', 'Texture Cos Closer']

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
            anchor_output = embeddings[anchor]
            shape_output = embeddings[shape_match]
            texture_output = embeddings[texture_match]

            # Retrieve similarities if they've already been calculated
            if (anchor, shape_match) in sim_dict.keys() or (shape_match, anchor) in sim_dict.keys():
                try:
                    shape_dot = sim_dict[(anchor, shape_match)][0]
                    shape_cos = sim_dict[(anchor, shape_match)][1]
                except KeyError:
                    shape_dot = sim_dict[(shape_match, anchor)][0]
                    shape_cos = sim_dict[(shape_match, anchor)][1]
            else:
                shape_dot = np.dot(anchor_output, shape_output)
                shape_cos = spatial.distance.cosine(anchor_output, shape_output)
                sim_dict[(anchor, shape_match)] = [shape_dot, shape_cos]

            if (anchor, texture_match) in sim_dict.keys() or (texture_match, anchor) in sim_dict.keys():
                try:
                    texture_dot = sim_dict[(anchor, texture_match)][0]
                    texture_cos = sim_dict[(anchor, texture_match)][1]
                except KeyError:
                    texture_dot = sim_dict[(texture_match, anchor)][0]
                    texture_cos = sim_dict[(texture_match, anchor)][1]
            else:
                texture_dot = np.dot(anchor_output, texture_output)
                texture_cos = spatial.distance.cosine(anchor_output, texture_output)
                sim_dict[(anchor, texture_match)] = [texture_dot, texture_cos]

            if verbose:
                print("For " + anchor + " paired with " + shape_match + ", " + texture_match + ":")
                print("\tShape match dot product: " + str(shape_dot))
                print("\tShape match cos similarity: " + str(shape_cos))
                print("\t-------------")
                print("\tTexture match dot: " + str(texture_dot))
                print("\tTexture match cos similarity: " + str(texture_cos))
                print()

            df.at[i, 'Shape Dot'] = shape_dot
            df.at[i, 'Shape Cos'] = shape_cos
            df.at[i, 'Texture Dot'] = texture_dot
            df.at[i, 'Texture Cos'] = texture_cos

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

        df.to_csv('results/' + model_type + '/similarity/' + anchor[:-4] + '.csv', index=False)


def get_embeddings(dir, model, model_type):
    """ Retrieves embeddings for each image in a dataset from the penultimate
    layer of a given model. Stores the embeddings in a dictionary (indexed by
    image name, eg. cat4-truck3). Returns the dictionary and stores it in a json
    file (model_type_embeddings.json)

    :param dir: path of the dataset
    :param model: the model to extract the embeddings from
    :param model_type: the type of model, eg. saycam, resnet50, etc.

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
    dataset = GeirhosStyleTransferDataset(dir, '')
    num_images = dataset.__len__()

    softmax = nn.Softmax(dim=1)

    # Remove the final layer from the model
    if model_type == 'saycam' or model_type == 'saycamA' or model_type == 'saycamS'\
            or model_type == 'saycamY':
        modules = list(model.module.children())[:-1]
        penult_model = nn.Sequential(*modules)
    elif model_type == 'resnet50':
        modules = list(model.children())[:-1]
        penult_model = nn.Sequential(*modules)

    for p in penult_model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        # Iterate over images
        for i in range(num_images):
            im, name, shape, texture, shape_spec, texture_spec = dataset.__getitem__(i)
            im = im.unsqueeze(0)

            # Pass image into model
            embedding = softmax(penult_model(im)).numpy().squeeze()

            embedding_dict[name] = embedding.tolist()

    with open('embeddings/' + model_type + '_embeddings.json', 'w') as file:
        json.dump(embedding_dict, file)

    return embedding_dict


if __name__ == '__main__':
    """Passes images one at a time through a given model and stores/plots the results
    (the shape/texture of the image, the classification made, and whether or not
    the classifcation was a shape classification, a texture classification, or neither.)
    
    By default, the model is the SAYCAM-trained resnext model, and the dataset is the
    Geirhos ImageNet style-transfer dataset. These options can be changed when running
    this program in the terminal by using the -m and -d flags."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Example: saycam, resnet50', required=False, default='saycam')
    parser.add_argument('-v', '--verbose', help='Prints results.', required=False, action='store_true')
    parser.add_argument('-p', '--plot', help='Plots results.', required=False, action='store_true')
    parser.add_argument('-t', '--triplets', help='Obtains similarities for triplets of images.',
                        required=False, action='store_true')
    args = parser.parse_args()

    batch_size = 1
    shape_categories = sc.get_human_object_recognition_categories()  # list of 16 classes in the Geirhos style-transfer dataset
    shape_dir = 'stimuli-shape/style-transfer'
    texture_dir = 'stimuli-texture/style-transfer'
    plot = args.plot
    verbose = args.verbose
    t = args.triplets

    model_type = args.model  # 'saycam' or 'resnet50'

    if model_type == 'saycam':
        # Load Emin's pretrained SAYCAM model + ImageNet classifier from its .tar file
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        model = nn.DataParallel(model)
        checkpoint = torch.load('models/fz_IN_resnext50_32x4d_augmentation_True_SAY_5_288.tar',
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
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
    elif model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_type == 'vgg11':
        model = models.resnet18(pretrained=True)

    # Put model in evaluation mode
    model.eval()

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
        try:
            os.mkdir('results/' + model_type + '/similarity')
        except FileExistsError:
            pass

        try:
            os.mkdir('figures/' + model_type + '/similarity')
        except FileExistsError:
            pass

        try:
            embeddings = json.load(open('embeddings/' + model_type + '_embeddings.json'))
        except FileNotFoundError:
            embeddings = get_embeddings(shape_dir, model, model_type)

        #triplets(model_type, embeddings, verbose, shape_dir)
        #calculate_similarity_averages(model_type, shape_categories, plot)
        calculate_similarity_totals(model_type)

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
