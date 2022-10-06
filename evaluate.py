import pandas as pd
import glob
import os
from data import SilhouetteTriplets

# This defines the number of times triplets are randomly selected in the
# calculation of proportions. In other words, when num_draws = n, the
# shape bias proportion calculated for a given model is the average of
# calculated shape bias for n random draws of triplets. Note that these
# random draws are the same across models; the specific triplets to be
# selected are randomly chosen by the new_seed function in main.py and
# are stored. This ensures that all models see the same draws of random
# triplets. New random draws can be generated with the --new_seed flag.
num_draws = 3


def get_num_draws():
    """Returns the global variable num_draws; this function is used by the
    new_seed function in main.py, which generates num_draws random draws
    of triplets to be accessed by all models until a new seed is generated."""
    return num_draws


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


def calculate_totals(shape_categories, result_dir):
    """Calculates the total number of shape, texture, and neither shape nor
    texture decisions by Geirhos shape class (and overall). Stores these
    results in a CSV and optionally prints them out.

    :param shape_categories: a list of Geirhos shape classes.
    :param result_dir: where to store the results."""

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
        if filename[-4:] != '.csv' or filename == 'totals.csv' or filename == 'proportions_avg.csv':
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
        shape_idx = shape_categories.index(shape)
        result_df.at[shape_idx, 'Shape Category'] = shape
        result_df.at[shape_idx, 'Number Shape Decisions'] = shape_dict[shape]
        result_df.at[shape_idx, 'Number Texture Decisions'] = texture_dict[shape]
        result_df.at[shape_idx, 'Number Neither'] = neither_dict[shape]
        result_df.at[shape_idx, 'Number Restricted Shape Decisions'] = restricted_shape_dict[shape]
        result_df.at[shape_idx, 'Number Restricted Texture Decisions'] = restricted_texture_dict[shape]
        result_df.at[shape_idx, 'Total Number Stimuli'] = shape_dict[shape] + texture_dict[shape] +\
                                                          neither_dict[shape]

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


def calculate_proportions(model_type, result_dir):
    """Calculates the proportions of shape and texture decisions for a given model.
    There are two proportions calculated for both shape and texture: 1) with neither
    shape nor texture decisions included, and 2) without considering 'neither'
    decisions. Stores these proportions in a text file and optionally prints them.

    :param model_type: saycam, resnet50, etc.
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

    columns = ['Model', 'Metric', 'Shape Match Closer', 'Texture Match Closer']
    metrics = ['no_neither', 'neither', 'restricted']
    results = pd.DataFrame(index=range(len(metrics)), columns=columns)

    results.at[:, 'Model'] = model_type
    for i in range(len(metrics)):
        results.at[i, 'Metric'] = metrics[i]
    results.at[0, 'Shape Match Closer'] = shape_texture
    results.at[0, 'Texture Match Closer'] = texture_shape
    results.at[1, 'Shape Match Closer'] = shape_all
    results.at[1, 'Texture Match Closer'] = texture_all
    results.at[2, 'Shape Match Closer'] = shape_restricted
    results.at[2, 'Texture Match Closer'] = texture_restricted

    '''
    strings = ["Proportion of shape decisions (disregarding 'neither' decisions): " + str(shape_texture),
               "Proportion of texture decisions (disregarding 'neither' decisions): " + str(texture_shape),
               "Proportion of shape decisions (including 'neither' decisions): " + str(shape_all),
               "Proportion of texture decisions (including 'neither' decisions): " + str(texture_all),
               "Proportion of shape decisions (restricted to only shape/texture classes): " + str(shape_restricted),
               "Proportion of texture decisions (restricted to only shape/texture classes): " + str(texture_restricted)]
    file = open(result_dir + '/proportions.txt', 'w')

    for i in range(len(strings)):
        file.write(strings[i] + '\n')

    file.close()
    '''
    results.to_csv(result_dir + '/proportions_avg.csv', index=False)


def calculate_similarity_totals(args, model_type, stimuli_dir, n=-1, N=0):
    """Calculates proportion of times the shape/texture dot product/cosine similarity/
    Euclidean distance is closer for a given model. Stores proportions as a csv.

    :param args: command line args to be obtained from main.py
    :param model_type: saycam, resnet50, etc.
    :param stimuli_dir: location of stimuli that the results are calculated for
    :param n: for use when model_type = resnet50_random or ViTB16_random. Specifies
              which particular random model to use
    :param N: total number of random models
    """

    if args.bg_match:
        calculate_similarity_totals_bg_match(args, model_type, stimuli_dir, n=-1, N=0)
        return

    if args.novel:
        num_draws = 1
    else:
        num_draws = get_num_draws()

    metrics = ['cos', 'dot', 'ed']

    if 'random' in model_type and n != -1:
        model_type = '{0}_{1}'.format(model_type, n)

    dataset = SilhouetteTriplets(args, stimuli_dir, None)
    sim_dir = 'results/{0}/{1}'.format(model_type, stimuli_dir)

    # Values for dictionary below: [shape_closer, texture_closer]
    results_by_metric_total = {key: [0, 0] for key in metrics}

    columns = ['Model', 'Metric', 'Shape Match Closer', 'Texture Match Closer']

    results = pd.DataFrame(index=range(len(metrics)), columns=columns)
    results.loc[:, 'Model'] = model_type

    if 'random' in model_type and n == -1:  # Compute averages across random models
        for i in range(1, N+1):
            random_df = pd.read_csv('results/{0}_{1}/{2}/proportions_avg.csv'.format(model_type,
                                                                                     i,
                                                                                     stimuli_dir))
            for index, row in random_df.iterrows():
                metric = row['Metric']

                shape_closer = row['Shape Match Closer']
                texture_closer = row['Texture Match Closer']

                results_by_metric_total[metric][0] += shape_closer
                results_by_metric_total[metric][1] += texture_closer

        for i in range(len(metrics)):
            metric = metrics[i]
            metric_results = results_by_metric_total[metric]  # [shape_closer, texture_closer, color_closer]

            results.at[i, 'Metric'] = metric
            results.at[i, 'Shape Match Closer'] = metric_results[0] / N
            results.at[i, 'Texture Match Closer'] = metric_results[1] / N

        results.to_csv('{0}/proportions_avg.csv'.format(sim_dir), index=False)
        return

    for random_draw in range(num_draws):
        selected_triplets = dataset.select_capped_triplets(random_draw)
        num_rows = 0
        results_by_metric = {key: [0, 0] for key in metrics}

        for file in glob.glob('{0}/*.csv'.format(sim_dir)):
            if file == '{0}/averages.csv'.format(sim_dir) or file == '{0}/proportions.csv'.format(sim_dir)\
                    or file == '{0}/matrix.csv'.format(sim_dir) or file == '{0}/proportions_avg.csv'.format(sim_dir)\
                    or file == '{0}/proportions_OLD.csv'.format(sim_dir):
                continue

            df = pd.read_csv(file)
            selection = selected_triplets[file.split('/')[-1].replace('csv', 'png')]
            df2 = pd.DataFrame(columns=df.columns, index=range(dataset.max_num_triplets()))
            df2_idx = 0

            for index, row in df.iterrows():
                triplet = ['{0}.png'.format(row['Anchor']), '{0}.png'.format(row['Shape Match']),
                           '{0}.png'.format(row['Texture Match'])]

                if triplet in selection:
                    df2.loc[df2_idx, :] = row[:]
                    df2_idx += 1

            for index, row in df2.iterrows():
                metric = row['Metric']

                shape_closer = int(row['Shape Match Closer'])
                texture_closer = int(row['Texture Match Closer'])

                results_by_metric[metric][0] += shape_closer
                results_by_metric[metric][1] += texture_closer

                num_rows += 1

        num_rows = num_rows // len(metrics)  # Each triplet appears len(metric) times.
        for i in range(len(metrics)):
            for j in range(2):
                results_by_metric_total[metrics[i]][j] += results_by_metric[metrics[i]][j] / num_rows

    for i in range(len(metrics)):
        metric = metrics[i]
        metric_results = results_by_metric_total[metric]  # [shape_closer, texture_closer, color_closer]

        results.at[i, 'Metric'] = metric
        results.at[i, 'Shape Match Closer'] = metric_results[0] / num_draws
        results.at[i, 'Texture Match Closer'] = metric_results[1] / num_draws

    results.to_csv('{0}/proportions_avg.csv'.format(sim_dir), index=False)


def calculate_similarity_totals_bg_match(args, model_type, stimuli_dir, n=-1, N=0):
    if args.novel:
        num_draws = 1
    else:
        num_draws = get_num_draws()

    metrics = ['cos', 'dot', 'ed']

    if 'random' in model_type and n != -1:
        model_type = '{0}_{1}'.format(model_type, n)

    dataset = SilhouetteTriplets(args, stimuli_dir.split('/')[-1], None)
    sim_dir = 'results/{0}/{1}'.format(model_type, stimuli_dir)

    # Values for dictionary below: [shape_closer, texture_closer]
    results_by_metric_total = {key: [0, 0, 0] for key in metrics}

    columns = ['Model', 'Metric', 'Shape Match Closer', 'Texture Match Closer', 'BG Match Closer']

    results = pd.DataFrame(index=range(len(metrics)), columns=columns)
    results.loc[:, 'Model'] = model_type

    if 'random' in model_type and n == -1:  # Compute averages across random models
        for i in range(1, N+1):
            random_df = pd.read_csv('results/{0}_{1}/{2}/proportions_avg.csv'.format(model_type,
                                                                                     i,
                                                                                     stimuli_dir))
            for index, row in random_df.iterrows():
                metric = row['Metric']

                shape_closer = row['Shape Match Closer']
                texture_closer = row['Texture Match Closer']
                bg_closer = row['BG Match Closer']

                results_by_metric_total[metric][0] += shape_closer
                results_by_metric_total[metric][1] += texture_closer
                results_by_metric_total[metric][2] += bg_closer

        for i in range(len(metrics)):
            metric = metrics[i]
            metric_results = results_by_metric_total[metric]  # [shape_closer, texture_closer, color_closer]

            results.at[i, 'Metric'] = metric
            results.at[i, 'Shape Match Closer'] = metric_results[0] / N
            results.at[i, 'Texture Match Closer'] = metric_results[1] / N
            results.at[i, 'BG Match Closer'] = metric_results[2] / N

        results.to_csv('{0}/proportions_avg.csv'.format(sim_dir), index=False)
        return

    for random_draw in range(1):
        num_rows = 0
        results_by_metric = {key: [0, 0, 0] for key in metrics}

        for file in glob.glob('{0}/*.csv'.format(sim_dir)):
            if file == '{0}/averages.csv'.format(sim_dir) or file == '{0}/proportions.csv'.format(sim_dir)\
                    or file == '{0}/matrix.csv'.format(sim_dir) or file == '{0}/proportions_avg.csv'.format(sim_dir)\
                    or file == '{0}/proportions_OLD.csv'.format(sim_dir):
                continue

            df = pd.read_csv(file)

            for index, row in df.iterrows():
                metric = row['Metric']

                shape_closer = int(row['Shape Match Closer'])
                texture_closer = int(row['Texture Match Closer'])
                bg_closer = int(row['BG Match Closer'])

                results_by_metric[metric][0] += shape_closer
                results_by_metric[metric][1] += texture_closer
                results_by_metric[metric][2] += bg_closer

                num_rows += 1

        num_rows = num_rows // len(metrics)  # Each triplet appears len(metric) times.
        for i in range(len(metrics)):
            for j in range(3):
                results_by_metric_total[metrics[i]][j] += results_by_metric[metrics[i]][j] / num_rows

    for i in range(len(metrics)):
        metric = metrics[i]
        metric_results = results_by_metric_total[metric]  # [shape_closer, texture_closer, color_closer]

        results.at[i, 'Metric'] = metric
        results.at[i, 'Shape Match Closer'] = metric_results[0]
        results.at[i, 'Texture Match Closer'] = metric_results[1]
        results.at[i, 'BG Match Closer'] = metric_results[2] 

    results.to_csv('{0}/proportions_avg.csv'.format(sim_dir), index=False)
