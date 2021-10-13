import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import json
import numpy as np


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
    :param texture: the texture classification of the given image.
    :param model_type: saycam, resnet50, etc."""

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


def plot_similarity_histograms(model_type, g):
    """First plots 6 regular histograms: one set of 2 for cosine similarity between anchor
    images and shape/texture matches, one set of 2 for dot product between anchor images
    and shape/texture matches, and one set of 2 for Euclidean distance between anchor images
    and shape/texture matches (across all classes). Then plots a set of two "difference"
    histograms, which plot the difference between the cosine similarity, dot product, or
    Euclidean distance to the shape match and texture match (eg. cos_difference =
    cos_similarity(anchor, shape match) - cos_similarity(anchor, texture match)).

    :param model_type: saycam, resnet50, etc.
    :param g: true if using the grayscale Geirhos dataset
    """

    # NOTE: THIS CODE NEEDS TO BE MADE COMPATIBLE WITH THE NEW CSV FORMAT. DO NOT RUN YET

    # Create directory
    if g:
        plot_dir = 'figures/' + model_type + '/grayscale'
    else:
        plot_dir = 'figures/' + model_type + '/similarity'
    try:
        os.mkdir(plot_dir)
    except FileExistsError:
        pass
    plot_dir += '/'

    # Collect data
    sim_dir = 'results/' + model_type + '/similarity/'

    shape_dot = []
    shape_cos = []
    shape_ed = []
    texture_dot = []
    texture_cos = []
    texture_ed = []
    cos_difference = []
    dot_difference = []
    ed_difference = []

    for file in glob.glob(sim_dir + '*.csv'):

        if file == sim_dir + 'averages.csv' or file == sim_dir + 'proportions.csv' \
                or file == sim_dir + 'matrix.csv':
            continue
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            shape_dot.append(float(row['Shape Dot']))
            shape_cos.append(float(row['Shape Cos']))
            shape_ed.append(float(row['Shape ED']))
            texture_dot.append(float(row['Texture Dot']))
            texture_cos.append(float(row['Texture Cos']))
            texture_ed.append(float(row['Texture ED']))

            cos_difference.append(float(row['Shape Cos']) - float(row['Texture Cos']))
            dot_difference.append(float(row['Shape Dot']) - float(row['Texture Dot']))
            ed_difference.append(float(row['Shape ED']) - float(row['Texture ED']))

    # Plot regular histograms
    fig, axs = plt.subplots(3, 2)
    fig.set_figheight(14)
    fig.set_figwidth(12)
    plt.suptitle('Histogram of Similarities Between Anchor Images & Shape/Texture Matches',
                 fontsize='xx-large')

    y1, x1, _ = axs[0, 0].hist(shape_cos, color='#ffb4b4', bins=30)
    axs[0, 0].set_title('Cosine Similarity: Shape Match')

    y2, x2, _ = axs[0, 1].hist(texture_cos, color='#ff7694', bins=30)
    axs[0, 1].set_ylim([0, max(y1.max(), y2.max() + 1000)])
    axs[0, 0].set_ylim([0, max(y1.max(), y2.max()) + 1000])
    axs[0, 1].set_title('Cosine Similarity: Texture Match')

    y3, x3, _ = axs[1, 0].hist(shape_dot, color='#ba6ad0', bins=30)
    axs[1, 0].set_title('Dot Product: Shape Match')

    y4, x4, _ = axs[1, 1].hist(texture_dot, color='#645f97', bins=30)
    axs[1, 0].set_ylim([0, max(y3.max(), y4.max()) + 1000])
    axs[1, 1].set_ylim([0, max(y3.max(), y4.max()) + 1000])
    axs[1, 1].set_title('Dot Product: Texture Match')

    y5, x5, _ = axs[2, 0].hist(shape_ed, color='#fee8ca', bins=30)
    axs[2, 0].set_title('Euclidean Distance: Shape Match')

    y6, x6, _ = axs[2, 1].hist(texture_ed, color='#cb8f32', bins=30)
    axs[2, 0].set_ylim([0, max(y5.max(), y6.max()) + 1000])
    axs[2, 1].set_ylim([0, max(y5.max(), y6.max()) + 1000])
    axs[2, 1].set_title('Euclidean Distance: Texture Match')

    plt.savefig(plot_dir + 'regular.png')

    # Plot difference histograms
    fig, axs = plt.subplots(1, 3)
    fig.set_figheight(6)
    fig.set_figwidth(18)
    plt.suptitle('Histogram of Difference between Shape & Texture Match Similarities', fontsize='xx-large')

    y7, x7, _ = axs[0].hist(cos_difference, color='#ff7694', bins=30)
    axs[0].set_title('Cosine Similarity Difference: Shape - Texture')

    y8, x8, _ = axs[1].hist(dot_difference, color='#ffb4b4', bins=30)
    axs[1].set_title('Dot Product Difference: Shape - Texture')

    y9, x9, _ = axs[2].hist(ed_difference, color='#fee8ca', bins=30)
    axs[2].set_title('Euclidean Distance Difference: Shape - Texture')

    plt.savefig(plot_dir + 'difference.png')


def plot_norm_histogram(model_type, c, g):
    """Plots a histogram of embedding norms for a given model.

    :param model_type: resnet40, saycam, etc.
    :param c: True if using the artificial/cartoon stimuli dataset.
    :param g: True if using the grayscale Geirhos dataset."""

    # Create directory
    plot_dir = 'figures/' + model_type + '/embeddings'

    try:
        os.mkdir(plot_dir)
    except FileExistsError:
        pass

    plot_dir += '/norms'

    try:
        os.mkdir(plot_dir)
    except FileExistsError:
        pass

    plot_dir += '/'
    b = 30  # Number of bins

    # Load embeddings
    if c:
        embedding_dir = 'embeddings/' + model_type + '_cartoon.json'
        fig_dir = plot_dir + model_type + '_cartoon.png'
        b = 30
    elif g:
        embedding_dir = 'embeddings/' + model_type + '_gray.json'
        fig_dir = plot_dir + model_type + '_gray.png'
    else:
        embedding_dir = 'embeddings/' + model_type + '_embeddings.json'
        fig_dir = plot_dir + model_type + '.png'

    embeddings = json.load(open(embedding_dir))

    norms = []

    for image in embeddings.keys():
        e = embeddings[image]
        norms.append(np.linalg.norm(e))

    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    fig.set_figheight(6)
    fig.set_figwidth(9)

    y, x, _ = ax.hist(norms, color='#ff7694', bins=b)
    plt.title('Histogram of Embedding Norms: ' + model_type)
    plt.tight_layout()

    plt.savefig(fig_dir)


def plot_similarity_bar(g, c):
    """Plots a stacked bar plot of proportion shape/texture/(color) match according to
    similarity across models.

    :param g: True if using the grayscale Geirhos dataset.
    :param c: True if using the artificial/cartoon stimuli dataset."""

    model_types = ['dino_resnet50', 'mocov2', 'swav', 'clipRN50', 'clipRN50x4', 'clipRN50x16',
                   'clipViTB32', 'clipViTB16', 'alexnet', 'vgg16', 'resnet50', 'saycam', 'saycamA',
                   'saycamS', 'saycamY']
    model_labels = ['DINO-ResNet50', 'MoCoV2-ResNet50', 'SwAV-ResNet50', 'CLIP-ResNet50',
                    'CLIP-ResNet50x4', 'CLIP-ResNet50x16', 'CLIP-ViTB/32', 'CLIP-ViTB/16', 'AlexNet',
                    'VGG-16', 'ResNet-50', 'ImageNet SAYCAM', 'SAYCAM-A', 'SAYCAM-S', 'SAYCAM-Y']

    if c:
        sub = 'cartoon'
        d = 3
    elif g:
        sub = 'grayscale'
        d = 2
    else:
        sub = 'similarity'
        d = 2

    proportions = {key: np.zeros((3, d)) for key in model_types}  # Rows: cos,dot,ed. Columns: shape,text,color

    for model in model_types:
        df = pd.read_csv('results/' + model + '/' + sub + '/proportions_avg.csv')

        cos_row = df.loc[df['Metric'] == 'cos']
        dot_row = df.loc[df['Metric'] == 'dot']
        ed_row = df.loc[df['Metric'] == 'ed']

        sim_cos = [float(cos_row['Shape Match Closer']), float(cos_row['Texture Match Closer'])]
        sim_dot = [float(dot_row['Shape Match Closer']), float(dot_row['Texture Match Closer'])]
        sim_ed = [float(ed_row['Shape Match Closer']), float(ed_row['Texture Match Closer'])]

        if c:
            sim_cos.append(float(cos_row['Color Match Closer']))
            sim_dot.append(float(dot_row['Color Match Closer']))
            sim_ed.append(float(ed_row['Color Match Closer']))

        proportions[model][0, :] = sim_cos
        proportions[model][1, :] = sim_ed
        proportions[model][2, :] = sim_dot

    if c:
        measures = [['shape cos', 'texture cos', 'color cos'],
                    ['shape ed', 'texture ed', 'color ed'],
                    ['shape dot', 'texture dot', 'color dot']]
        measures2 = ['shape cos', 'texture cos', 'color cos',
                     'shape ed', 'texture ed', 'color ed',
                     'shape dot', 'texture dot', 'color dot']
    else:
        measures = [['shape cos', 'texture cos'],
                    ['shape ed', 'texture ed'],
                    ['shape dot', 'texture dot']]
        measures2 = ['shape cos', 'texture cos',
                     'shape ed', 'texture ed',
                     'shape dot', 'texture dot']

    proportions2 = {key: np.zeros(len(model_types)).tolist() for key in measures2}
    for i in range(3):
        for j in range(len(model_types)):
            model = model_types[j]
            m = measures[i]
            for k in range(len(m)):
                proportions2[m[k]][j] = proportions[model][i, k]

    pdf = pd.DataFrame(proportions2, index=model_labels)

    plt.style.use('ggplot')

    colors = [['#345eeb', '#1b6600', '#8f178d'], ['#4aace8', '#4cb825', '#cc43de'], ['#81d0f0', '#a3eb5b', '#eca1ed']]
    distance_metrics = [['Cosine Similarity', 'cos'], ['Euclidean Distance', 'ed'], ['Dot Product', 'dot']]

    for i in range(3):
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 9)
        ax.figure.set_size_inches(9.7, 5.3)

        pdf[measures[i]].plot.barh(stacked=True, color=colors[i], width=0.8, ax=ax)
        plt.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))  # bbox_to_anchor=(1.1, 1.05)
        plt.title('Proportions of Similarity Matches by Model: ' + distance_metrics[i][0])
        plt.tight_layout()
        plt.xlim([0, 1])
        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.xlabel('Proportion', size=13, color='black')
        plt.ylabel('Model', size=13, color='black')
        plt.savefig('figures/proportions_' + sub + '_' + distance_metrics[i][1] + '.png', bbox_inches = "tight")


def plot_number_of_triplets(c):
    """Plots a histogram of the number of triplets that can be formed using a given
    image as an anchor. The plots are organized by anchor image shape, texture, and color
    classes."""

    shape_classes = json.load(open('geirhos_shape_classes.json'))
    triplets = json.load(open('geirhos_triplets.json'))
    anchors = shape_classes.keys()

    frequencies = {key: 0 for key in anchors}

    colors = ['#f6eeeb', '#ead5cf', '#c8e0e4', '#7197a0', '#223053', '#755b99', '#fc6c4f', '#bb0a65',
              '#a4da56', '#311f31', '#d9d2e9', '#960206', '#cb020c', '#5b1c1e', '#3a542f', '#a36b32']
    classes_to_colors = {key.split('/')[2]: None for key in glob.glob('stimuli-shape/style-transfer/*/')}
    shape_class_frequencies = {key: 0 for key in classes_to_colors.keys()}
    texture_class_frequencies = {key: 0 for key in classes_to_colors.keys()}

    for i in range(len(colors)):
        c = list(classes_to_colors.keys())[i]
        classes_to_colors[c] = colors[i]

    for anchor in anchors:
        anchor_triplets = triplets[anchor]['triplets']
        frequencies[anchor] += len(anchor_triplets)
        shape_class_frequencies[shape_classes[anchor]['shape']] += len(anchor_triplets)
        texture_class_frequencies[shape_classes[anchor]['texture']] += len(anchor_triplets)

    shape_height = []
    texture_height = []

    shape_class_height = []
    texture_class_height = []

    color_shape = []
    color_texture = []

    color_class = []

    anchors_by_shape = sorted(anchors)
    anchors_by_texture = sorted(anchors, key=lambda anchor: anchor.split('-')[1][:-4])

    for anchor in anchors_by_shape:
        if frequencies[anchor] == 0:
            continue
        shape_height.append(frequencies[anchor])
        color_shape.append(classes_to_colors[shape_classes[anchor]['shape']])

    for anchor in anchors_by_texture:
        if frequencies[anchor] == 0:
            continue
        texture_height.append(frequencies[anchor])
        color_texture.append(classes_to_colors[shape_classes[anchor]['texture']])

    for c in sorted(list(classes_to_colors.keys())):
        shape_class_height.append(shape_class_frequencies[c])
        texture_class_height.append(texture_class_frequencies[c])
        color_class.append(classes_to_colors[c])

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    ax.get_xaxis().set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=classes_to_colors[label]) for label in sorted(list(classes_to_colors.keys()))]
    plt.legend(handles, sorted(list(classes_to_colors.keys())), loc='best')
    plt.bar(anchors_by_shape, shape_height, color=color_shape)
    plt.title('Number of Triplets per Anchor Image, Organized by Shape of the Anchor')
    plt.ylabel('Number of triplets')
    plt.xlabel('Anchor image')
    plt.tight_layout()
    plt.savefig('figures/anchors_by_shape.png')

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    ax.get_xaxis().set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=classes_to_colors[label]) for label in
               sorted(list(classes_to_colors.keys()))]
    plt.legend(handles, sorted(list(classes_to_colors.keys())), loc='best')
    plt.bar(anchors_by_texture, texture_height, color=color_shape)
    plt.title('Number of Triplets per Anchor Image, Organized by Texture of the Anchor')
    plt.ylabel('Number of triplets')
    plt.xlabel('Anchor image')
    plt.tight_layout()
    plt.savefig('figures/anchors_by_texture.png')

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    ax.get_xaxis().set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=classes_to_colors[label]) for label in
               sorted(list(classes_to_colors.keys()))]
    plt.legend(handles, sorted(list(classes_to_colors.keys())), loc='best')
    plt.bar(sorted(list(classes_to_colors.keys())), shape_class_height, color=color_class)
    plt.title('Number of Triplets per Anchor Image Shape Class')
    plt.ylabel('Number of triplets')
    plt.xlabel('Anchor image class')
    plt.tight_layout()
    plt.savefig('figures/anchors_by_shape_class.png')

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    ax.get_xaxis().set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=classes_to_colors[label]) for label in
               sorted(list(classes_to_colors.keys()))]
    plt.legend(handles, sorted(list(classes_to_colors.keys())), loc='best')
    plt.bar(sorted(list(classes_to_colors.keys())), texture_class_height, color=color_class)
    plt.title('Number of Triplets per Anchor Image Texture Class')
    plt.ylabel('Number of triplets')
    plt.xlabel('Anchor image class')
    plt.tight_layout()
    plt.savefig('figures/anchors_by_texture_class.png')




if __name__ == "__main__":
    plot_number_of_triplets(False)