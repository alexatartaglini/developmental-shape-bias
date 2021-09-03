import pandas as pd
import glob
import os


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


def calculate_similarity_totals(model_type, c, g, matrix=False):
    """Calculates proportion of times the shape/texture dot product/cosine similarity
    is closer for a given model. Stores proportions as a csv.

    :param model_type: saycam, resnet50, etc.
    :param c: true if the artificial/cartoon stimulus dataset is being used.
    :param g: true if grayscale dataset is being used.
    :param matrix: true if you want to calculate a matrix of totals instead of
                   proportions."""

    if c:
        sim_dir = 'results/' + model_type + '/cartoon/'
    elif g:
        sim_dir = 'results/' + model_type + '/grayscale/'
    else:
        sim_dir = 'results/' + model_type +'/similarity/'

    if not matrix:
        if c:
            columns = ['Model', 'Shape Cos Closer', 'Texture Cos Closer', 'Color Cos Closer',
                       'Shape Dot Closer', 'Texture Dot Closer', 'Color Dot Closer',
                       'Shape ED Closer', 'Texture ED Closer', 'Color ED Closer']
        else:
            columns = ['Model', 'Shape Dot Closer', 'Shape Cos Closer', 'Texture Dot Closer', 'Texture Cos Closer',
                   'Shape ED Closer', 'Texture ED Closer']
        results = pd.DataFrame(index=range(1), columns=columns)

        shape_dot = 0
        shape_cos = 0
        shape_ed = 0
        texture_dot = 0
        texture_cos = 0
        texture_ed = 0
        color_dot = 0
        color_cos = 0
        color_ed = 0
        num_rows = 0

    else:
        # Matrix: [0, 0] = shape match w/ dot and shape match w/ cos
        # [0, 1] = texture match w/ dot and shape match w/ cos
        # [1, 0] = shape match w/ dot and texture match w/ cos
        # [1, 1] = texture match w/ dot and texture match w/ cos
        columns = ['Model', ' ', 'Shape Match with Dot Product', 'Texture Match with Dot Product']
        results = pd.DataFrame(index=range(2), columns=columns)
        results.at[0, ' '] = 'Shape Match with Cosine Similarity'
        results.at[1, ' '] = 'Texture Match with Cosine Similarity'

        m0_0 = 0
        m0_1 = 0
        m1_0 = 0
        m1_1 = 0

    results.at[:, 'Model'] = model_type

    for file in glob.glob(sim_dir + '*.csv'):

        if file == sim_dir + 'averages.csv' or file == sim_dir + 'proportions.csv'\
                or file == sim_dir + 'matrix.csv':
            continue
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            shape_cos_closer = int(row['Shape Cos Closer'])
            shape_dot_closer = int(row['Shape Dot Closer'])
            shape_ed_closer = int(row['Shape ED Closer'])

            texture_cos_closer = int(row['Texture Cos Closer'])
            texture_dot_closer = int(row['Texture Dot Closer'])
            texture_ed_closer = int(row['Texture ED Closer'])

            if c:
                color_cos_closer = int(row['Color Cos Closer'])
                color_dot_closer = int(row['Color Dot Closer'])
                color_ed_closer = int(row['Color ED Closer'])

            if not matrix:
                shape_cos += shape_cos_closer
                shape_dot += shape_dot_closer
                shape_ed += shape_ed_closer
                texture_cos += texture_cos_closer
                texture_dot += texture_dot_closer
                texture_ed += texture_ed_closer
                if c:
                    color_cos += color_cos_closer
                    color_dot += color_dot_closer
                    color_ed += color_ed_closer
                num_rows += 1

            else:
                if shape_dot_closer == 1:
                    if shape_cos_closer == 1:
                        m0_0 += 1
                    elif texture_cos_closer == 1:
                        m1_0 += 1
                elif texture_dot_closer == 1:
                    if shape_cos_closer == 1:
                        m0_1 += 1
                    elif texture_cos_closer == 1:
                        m1_1 += 1

    if not matrix:
        results.at[0, 'Shape Cos Closer'] = shape_cos / num_rows
        results.at[0, 'Shape Dot Closer'] = shape_dot / num_rows
        results.at[0, 'Shape ED Closer'] = shape_ed / num_rows

        results.at[0, 'Texture Cos Closer'] = texture_cos / num_rows
        results.at[0, 'Texture Dot Closer'] = texture_dot / num_rows
        results.at[0, 'Texture ED Closer'] = texture_ed / num_rows

        if c:
            results.at[0, 'Color Cos Closer'] = color_cos / num_rows
            results.at[0, 'Color Dot Closer'] = color_dot / num_rows
            results.at[0, 'Color ED Closer'] = color_ed / num_rows

        results.to_csv(sim_dir + 'proportions.csv', index=False)
    else:
        results.at[0, 'Shape Match with Dot Product'] = m0_0
        results.at[1, 'Shape Match with Dot Product'] = m1_0
        results.at[0, 'Texture Match with Dot Product'] = m0_1
        results.at[1, 'Texture Match with Dot Product'] = m1_1

        results.to_csv(sim_dir + 'matrix.csv', index=False)


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