import pandas as pd
import glob
import os
from data import GeirhosTriplets, CartoonStimTrials, SilhouetteTriplets


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


def calculate_similarity_totals(model_type, c, s):
    """Calculates proportion of times the shape/texture dot product/cosine similarity/
    Euclidean distance is closer for a given model. Stores proportions as a csv.

    :param model_type: saycam, resnet50, etc.
    :param c: true if the artificial/cartoon stimulus dataset is being used.
    :param s: true if silhouette variant of style transfer dataset is being used.
    """

    num_draws = 3

    if c:
        sim_dir = 'results/' + model_type + '/cartoon/'
        dataset = CartoonStimTrials(None)
    elif s:
        sim_dir = 'results/' + model_type + '/silhouette/'
        dataset = SilhouetteTriplets(None)
    else:
        sim_dir = 'results/' + model_type +'/similarity/'
        dataset = GeirhosTriplets(None)

    metrics = ['cos', 'dot', 'ed']
    # Values for dictionary below: [shape_closer, texture_closer, color_closer]
    results_by_metric_total = {key: [0, 0, 0] for key in metrics}

    if c:
        columns = ['Model', 'Metric', 'Shape Match Closer', 'Texture Match Closer',
                   'Color Match Closer']
    else:
        columns = ['Model', 'Metric', 'Shape Match Closer', 'Texture Match Closer']

    results = pd.DataFrame(index=range(len(metrics)), columns=columns)
    results.at[:, 'Model'] = model_type

    for random_draw in range(num_draws):
        selected_triplets = dataset.select_capped_triplets()
        num_rows = 0
        results_by_metric = {key: [0, 0, 0] for key in metrics}

        for file in glob.glob(sim_dir + '*.csv'):
            if file == sim_dir + 'averages.csv' or file == sim_dir + 'proportions.csv'\
                    or file == sim_dir + 'matrix.csv' or file == sim_dir + 'proportions_avg.csv':
                continue

            df = pd.read_csv(file)
            selection = selected_triplets[file.split('/')[3].replace('csv', 'png')]
            df2 = pd.DataFrame(columns=df.columns, index=range(dataset.max_num_triplets()))
            df2_idx = 0

            for index, row in df.iterrows():
                if c:
                    triplet = [row['Anchor'], row['Shape Match'] + '.png', row['Texture Match'] + '.png',
                               row['Color Match'] + '.png']
                else:
                    triplet = [row['Anchor'] + '.png', row['Shape Match'] + '.png', row['Texture Match'] + '.png']

                if triplet in selection:
                    df2.loc[df2_idx, :] = row[:]
                    df2_idx += 1

            for index, row in df2.iterrows():
                metric = row['Metric']

                shape_closer = int(row['Shape Match Closer'])
                texture_closer = int(row['Texture Match Closer'])

                results_by_metric[metric][0] += shape_closer
                results_by_metric[metric][1] += texture_closer

                if c:
                    color_closer = int(row['Color Match Closer'])
                    results_by_metric[metric][2] += color_closer

                num_rows += 1

        num_rows = num_rows // len(metrics)  # Each triplet appears len(metric) times.
        for i in range(len(metrics)):
            for j in range(len(metrics[i])):
                results_by_metric_total[metrics[i]][j] += results_by_metric[metrics[i]][j] / num_rows

    for i in range(len(metrics)):
        metric = metrics[i]
        metric_results = results_by_metric_total[metric]  # [shape_closer, texture_closer, color_closer]

        results.at[i, 'Metric'] = metric
        results.at[i, 'Shape Match Closer'] = metric_results[0] / num_draws
        results.at[i, 'Texture Match Closer'] = metric_results[1] / num_draws

        if c:
            results.at[i, 'Color Match Closer'] = metric_results[2] / num_draws

    results.to_csv(sim_dir + 'proportions_avg.csv', index=False)


def shape_bias_rankings(simulation):
    """This function ranks the models in order of highest shape and texture (and color)
    bias. Stores the rankings in text files inside the results folder.

    :param simulation: the type of simulation to calculate ranks from. eg. similarity
                       for Geirhos triplets and cartoon for cartoon dataset quadruplets."""

    biases = ['Shape Match Closer', 'Texture Match Closer']
    bias_titles = {'Shape Match Closer': 'Shape Bias', 'Texture Match Closer': 'Texture Bias'}

    if simulation == 'cartoon':
        biases.append('Color Match Closer')
        color_bias_rank = ['/']
        bias_titles['Color Match Closer'] = 'Color Bias'

    metrics = ['cos', 'dot', 'ed']
    metric_titles = {'cos': 'Cosine Similarity', 'dot': 'Dot Product', 'ed': 'Euclidean Distance'}

    ranks = {rank: {metric: ['/'] for metric in metrics} for rank in biases}

    for bias in biases:
        for i in range(len(metrics)):
            bias_unsorted = []  # ranks for models that ARE eg. shape biased
            not_bias_unsorted = []  # ranks for models that ARE NOT eg. shape biased

            for model_dir in glob.glob('results/*/'):
                if model_dir == 'results/visualizations/':
                    continue
                biased = True

                model = model_dir.split('/')[1]
                props = pd.read_csv(model_dir + simulation + '/proportions_avg.csv')  # DF of proportions

                row = props.iloc[i, :]
                metric = row['Metric']
                val = row[bias]

                for other_bias in biases:
                    if val < row[other_bias]:
                        biased = False
                        not_bias_unsorted.append([model, val])
                        break
                if biased:
                    bias_unsorted.append([model, val])

            not_bias_sorted = sorted(not_bias_unsorted, key=lambda m: m[1])
            for i in range(len(not_bias_sorted)):
                not_bias_sorted[i] = not_bias_sorted[i][0]

            not_bias_sorted.reverse()

            bias_sorted = sorted(bias_unsorted, key=lambda m: m[1])
            for i in range(len(bias_sorted)):
                bias_sorted[i] = bias_sorted[i][0]

            if bias_sorted:
                for m in bias_sorted:
                    ranks[bias][metric].insert(0, m)

            if not_bias_sorted:
                for m in not_bias_sorted:
                    ranks[bias][metric].append(m)

    for metric in metrics:
        report_str = 'Model Rankings by {0}\n'.format(metric_titles[metric])
        report_str += '-----------------------------------------------------\n'

        for bias in biases:
            report_str += '-{0} Rankings:\n\n'.format(bias_titles[bias])
            rankings = ranks[bias][metric]
            j = 0

            for i in range(len(rankings)):
                m = rankings[i]
                if m == '/':
                    j = 1
                    report_str += '////(models above this line are {0}ed)////\n'.format(bias_titles[bias].lower())
                else:
                    report_str += '{0:<3} {1}\n'.format(str(i + 1 - j) + '.', m)
            report_str += '\n-----------------------------------------------------\n'

        with open('results/' + simulation + '_rankings_' + metric + '.txt', 'w') as f:
            f.write(report_str)
