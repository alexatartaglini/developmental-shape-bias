import torch
import torchvision.models as models
from torchvision import datasets, transforms, models
import PIL
import copy
import os
import numpy as np
import shutil
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


def csv_class_values(shape_dict, shape_categories):
    columns = ['Shape', 'Texture', 'Decision', 'Shape Category Value',
               'Texture Category Value', 'Decision Category Value',
               'Shape Decision', 'Texture Decision', 'Neither']

    for shape in shape_categories:
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

        df.to_csv('csv/' + shape + '.csv', index=False)


def calculate_totals(shape_categories):
    shape_dict = dict.fromkeys(shape_categories)
    texture_dict = dict.fromkeys(shape_categories)
    neither_dict = dict.fromkeys(shape_categories)
    for shape in shape_categories:
        shape_dict[shape] = 0
        texture_dict[shape] = 0
        neither_dict[shape] = 0

    for filename in os.listdir('csv'):
        df = pd.read_csv('csv/' + filename)
        shape = df['Shape'][0]
        for i, row in df.iterrows():
            shape_dict[shape] = shape_dict[shape] + row['Shape Decision']
            texture_dict[shape] += row['Texture Decision']
            neither_dict[shape] += row['Neither']

    for shape in shape_categories:
        print("Shape category: " + shape)
        print("\tNumber shape decisions: " + str(shape_dict[shape]))
        print("\tNumber texture decisions: " + str(texture_dict[shape]))
        print("\tNumber neither shape nor texture decisions: " + str(neither_dict[shape]))
        print()

    print("IN TOTAL:")
    print("\tNumber shape decisions: " + str(sum(shape_dict.values())))
    print("\tNumber texture decisions: " + str(sum(texture_dict.values())))
    print("\tNumber neither shape nor texture decisions: " + str(sum(neither_dict.values())))




if __name__ == '__main__':

    batch_size = 1
    shape_categories = sc.get_human_object_recognition_categories()  # list of 16 classes in the Geirhos style-transfer dataset
    shape_dir = 'stimuli-shape/style-transfer'
    texture_dir = 'stimuli-texture/style-transfer'
    plot = False
    verbose = True

    '''
    shape_dict = dict.fromkeys(shape_categories)  # for storing the results
    shape_categories0 = [shape + '0' for shape in shape_categories]
    shape_dict0 = dict.fromkeys(shape_categories0)
    for shape in shape_categories:
        shape_dict[shape] = shape_dict0

    shape_dict['airplane']['airplane'] = 0

    # Load Emin's pretrained SAYCAM model + ImageNet classifier from its .tar file
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('models/fz_IN_resnext50_32x4d_augmentation_True_SAY_5_288.tar',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

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
            print('decision for ' + im_dir + ': ' + decision)
            # for j in range(16):
            # print(shape_categories[j] + ': ' + str(class_values[j]))
            # print()
        if plot:
            plot_class_values(shape_categories, class_values, im_dir, shape, texture)

        shape_dict[shape][texture + '0'] = [decision, class_values]

    '''
    #csv_class_values(shape_dict, shape_categories)
    calculate_totals(shape_categories)
