import torch
from torchvision import datasets, transforms, models
import PIL
import copy
import os
import numpy as np
import argparse
import pandas as pd
import probabilities_to_decision
import helper.human_categories as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from dataloader import GeirhosStyleTransferDataset


def plot_class_values(categories, class_values, im, shape, texture):
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

    plt.savefig('figures/' + im)


def csv_class_values(shape_dict, shape_categories, csv_dir):
    """Writes the shape category, texture category, and model decision for all
    shape-texture combinations in a given Geirhos shape class to a CSV file.
    Also includes whether or not neither the shape or texture classification is made.

    :param shape_dict: a dictionary of values with shape category keys. Should
        store the decision made and the class values for a given shape-image combination.
    :param shape_categories: a list of all Geirhos shape classes.
    :param csv_dir: directory for storing the CSV."""

    columns = ['Shape', 'Texture', 'Decision', 'Shape Category Value',
               'Texture Category Value', 'Decision Category Value',
               'Shape Decision', 'Texture Decision', 'Neither']

    for shape in shape_categories:
        print(shape)
        df = pd.DataFrame(index=range(len(shape_categories)), columns=columns)
        df['Shape'] = shape

        for i, row in df.iterrows():
            texture = shape_categories[i]
            decision = shape_dict[shape][texture + '0'][0]
            class_values = shape_dict[shape][texture + '0'][1]

            row['Texture'] = texture
            row['Decision'] = decision
            row['Shape Category Value'] = class_values[shape_categories.index(shape)]
            row['Texture Category Value'] = class_values[shape_categories.index(texture)]
            row['Decision Category Value'] = class_values[shape_categories.index(decision)]

            row['Shape Decision'] = int(decision == shape)
            row['Texture Decision'] = int(decision == texture)
            row['Neither'] = int(decision != shape and decision != texture)

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

    columns = ['Shape Category', 'Number Shape Decisions', 'Number Texture Decisions',
               'Number Neither', 'Total Number Stimuli']
    result_df = pd.DataFrame(columns=columns, index=range(len(shape_categories) + 1))

    for shape in shape_categories:
        shape_dict[shape] = 0
        texture_dict[shape] = 0
        neither_dict[shape] = 0

    for filename in os.listdir(result_dir):
        if filename[-4:] != '.csv' or filename == 'totals.csv':
            continue

        df = pd.read_csv(result_dir + '/' + filename)
        shape = df['Shape'][0]
        for i, row in df.iterrows():
            shape_dict[shape] = shape_dict[shape] + row['Shape Decision']
            texture_dict[shape] += row['Texture Decision']
            neither_dict[shape] += row['Neither']

    for shape in shape_categories:
        if verbose:
            print("Shape category: " + shape)
            print("\tNumber shape decisions: " + str(shape_dict[shape]))
            print("\tNumber texture decisions: " + str(texture_dict[shape]))
            print("\tNumber neither shape nor texture decisions: " + str(neither_dict[shape]))
            print()

        shape_idx = shape_categories.index(shape)
        result_df.at[shape_idx, 'Shape Category'] = shape
        result_df.at[shape_idx, 'Number Shape Decisions'] = shape_dict[shape]
        result_df.at[shape_idx, 'Number Texture Decisions'] = texture_dict[shape]
        result_df.at[shape_idx, 'Number Neither'] = neither_dict[shape]
        result_df.at[shape_idx, 'Total Number Stimuli'] = shape_dict[shape] + texture_dict[shape] +\
                                                          neither_dict[shape]

    if verbose:
        print("IN TOTAL:")
        print("\tNumber shape decisions: " + str(sum(shape_dict.values())))
        print("\tNumber texture decisions: " + str(sum(texture_dict.values())))
        print("\tNumber neither shape nor texture decisions: " + str(sum(neither_dict.values())))

    idx = len(shape_categories)  # final row
    result_df.at[idx, 'Shape Category'] = 'total'
    result_df.at[idx, 'Number Shape Decisions'] = sum(shape_dict.values())
    result_df.at[idx, 'Number Texture Decisions'] = sum(texture_dict.values())
    result_df.at[idx, 'Number Neither'] = sum(neither_dict.values())
    result_df.at[idx, 'Total Number Stimuli'] = sum(neither_dict.values()) + \
                                                sum(texture_dict.values()) + sum(shape_dict.values())

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

    shape_texture = shape / (shape + texture)
    texture_shape = texture / (shape + texture)
    shape_all = shape / total
    texture_all = texture / total

    strings = ["Proportion of shape decisions (disregarding 'neither' decisions): " + str(shape_texture),
               "Proportion of texture decisions (disregarding 'neither' decisions): " + str(texture_shape),
               "Proportion of shape decisions (including 'neither' decisions): " + str(shape_all),
               "Proportion of shape decisions (including 'neither' decisions): " + str(texture_all)]
    file = open(result_dir + '/proportions.txt', 'w')

    for i in range(4):
        file.write(strings[i] + '\n')
        if verbose:
            print(strings[i])

    file.close()


if __name__ == '__main__':
    """Passes images one at a time through a given model and stores/plots the results
    (the shape/texture of the image, the classification made, and whether or not
    the classifcation was a shape classification, a texture classification, or neither.)
    
    By default, the model is the SAYCAM-trained resnext model, and the dataset is the
    Geirhos ImageNet style-transfer dataset. These options can be changed when running
    this program in the terminal by using the -m and -d flags."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Example: saycam, resnet50', required=False, default='saycam')

    batch_size = 1
    shape_categories = sc.get_human_object_recognition_categories()  # list of 16 classes in the Geirhos style-transfer dataset
    shape_dir = 'stimuli-shape/style-transfer'
    texture_dir = 'stimuli-texture/style-transfer'
    plot = True
    verbose = True

    model_type = 'resnet50'  # 'saycam' or 'resnet50'

    shape_dict = dict.fromkeys(shape_categories)  # for storing the results
    shape_categories0 = [shape + '0' for shape in shape_categories]
    shape_dict0 = dict.fromkeys(shape_categories0)
    for shape in shape_categories:
        shape_dict[shape] = shape_dict0

    if model_type == 'saycam':
        # Load Emin's pretrained SAYCAM model + ImageNet classifier from its .tar file
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('models/fz_IN_resnext50_32x4d_augmentation_True_SAY_5_288.tar',
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)

    # Load and process the images using my custom Geirhos style transfer dataset class
    g = GeirhosStyleTransferDataset(shape_dir, texture_dir)

    # Obtain ImageNet - Geirhos mapping
    mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
    softmax = torch.nn.Softmax(dim=1)

    # Pass images into the model one at a time
    for i in range(g.__len__()):
        im_dir, shape, texture, im = g.__getitem__(i)
        im = im.reshape(1, 3, 224, 224)

        output = model(im)
        soft_output = softmax(output).detach().numpy().squeeze()

        decision, class_values = mapping.probabilities_to_decision(soft_output)

        if verbose:
            print('Decision for ' + im_dir + ': ' + decision)
        if plot:
            plot_class_values(shape_categories, class_values, im_dir, shape, texture)

        shape_dict[shape][texture + '0'] = [decision, class_values]

    try:
        os.mkdir('results/' + model_type)
    except:
        pass

    csv_class_values(shape_dict, shape_categories, 'results/' + model_type)
    calculate_totals(shape_categories, 'results/' + model_type, verbose)
    calculate_proportions('results/' + model_type, verbose)

