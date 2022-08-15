import matplotlib.pyplot as plt
import pandas as pd


def get_model_list():
    return ['resnet50', 'resnet50_random', 'ViTB16', 'ViTB16_random',
                  'dino_resnet50', 'clipViTB16', 'saycamS']


def make_plots(args):
    """Creates plots according to command line arguments.

    :param args: command line arguments."""

    plot = args.plot

    if plot == 'alpha':
        plot_bias_vs_alpha(args)
    elif plot == 'size':
        plot_bias_vs_size(args)
    elif plot == 'alpha_random':
        plot_bias_vs_alpha(args, random=True)
    elif plot == 'size_random':
        plot_bias_vs_size(args, random=True)
    else:
        print('Plot type {0} not defined.'.format(plot))


def plot_bias_vs_alpha(args, random=False):
    """Plots shape bias proportions vs. alpha value (background saliency).

    :param args: command line arguments.
    :param random: true if plotting N random models + their average."""

    if random:
        model_list = ['{0}_{1}'.format(args.model, i) for i in range(1, args.N + 1)]
        model_list.insert(0, args.model)

        colors = ['#9c9c9c' for i in range(args.N)]
        colors.insert(0, '#16a1d9')

        labels = [str(i) for i in range(1, args.N + 1)]
        labels.insert(0, 'Average')

        plot_alphas = [0.4 for i in range(args.N)]
        plot_alphas.insert(0, (1))
    else:
        model_list = get_model_list()

        colors = ['#e274d0', '#e274d0', '#5ea6c6', '#5ea6c6',
                  '#57ad74', '#c69355', '#e6556c']
        labels = ['ResNet-50', 'ResNet-50 (Random)', 'ViT-B/16', 'ViT-B/16 (Random)',
                  'DINO ResNet-50', 'CLIP ViT-B/16', 'SAYCam-S']
        styles = ['solid', 'dashed', 'solid', 'dashed',
                  'solid', 'solid', 'solid']
        markers = ['o', 'o', '^', '^',
                   'o', '^', 'o']

    if args.novel:
        sim_dir = 'novel'
    else:
        sim_dir = 'geirhos'

    if args.classification:
        class_str = 'classifications/'
    else:
        class_str = ''

    if args.bg:
        bg_str = 'background_{0}/'.format(args.bg)
    else:
        bg_str = ''

    if random:
        if args.bg:
            bg_str_temp = bg_str[:-1]
        else:
            bg_str_temp = bg_str

        plot_dir = 'figures/{0}/{1}{2}'.format(args.model, class_str, bg_str_temp)
    elif not args.all_models:  # For now, this only works for models in model_list
        idx = model_list.index(args.model)
        colors = [colors[idx]]
        labels = [labels[idx]]
        styles = [styles[idx]]
        markers = [markers[idx]]
        model_list = [model_list[idx]]

        if args.bg:
            bg_str_temp = bg_str[:-1]
        else:
            bg_str_temp = bg_str

        plot_dir = 'figures/{0}/{1}{2}'.format(args.model, class_str, bg_str_temp)
    else:
        if args.bg:
            bg_str_temp = bg_str[:-1]
        else:
            bg_str_temp = bg_str

        plot_dir = 'figures/{0}{1}'.format(class_str, bg_str_temp)

    model_dict = {key: {"0.0": 0, "0.2": 0, "0.4": 0, "0.6": 0, "0.8": 0, "1": 0}
                  for key in model_list}

    for i in range(len(model_list)):
        model = model_list[i]

        for alpha in model_dict[model].keys():
            prop_dir = 'results/{0}/{1}{2}{3}-alpha{4}-size100-aligned/' \
                       'proportions_avg.csv'.format(model, class_str, bg_str,
                                                    sim_dir, alpha)

            props = pd.read_csv(prop_dir)
            shape_bias = props.at[0, 'Shape Match Closer']

            model_dict[model][alpha] = shape_bias

    alphas = list(model_dict[model].keys())

    plt.clf()
    plt.axhline(0.5, color='#808080', linestyle=(0, (1, 3)))  # Chance line

    for i in reversed(range(len(model_list))):
        model = model_list[i]
        if random:
            plt.plot(alphas, list(model_dict[model].values()), color=colors[i],
                     label=labels[i], marker='o', markersize=7.5,
                     markeredgecolor='black', markeredgewidth=0.5,
                     markerfacecolor=colors[i], alpha=plot_alphas[i])
        else:
            plt.plot(alphas, list(model_dict[model].values()), linestyle=styles[i],
                     color=colors[i], label=labels[i], marker=markers[i],
                     markersize=7.5, markeredgecolor='black', markeredgewidth=0.5,
                     markerfacecolor=colors[i])

    plt.title("Shape Bias vs. \u03B1")
    plt.xlabel('\u03B1 (Background Texture Transparency)', fontsize=10)
    plt.ylabel('Proportion of Shape Decisions', fontsize=10)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in reversed(range(len(model_list)))],
               [labels[idx] for idx in reversed(range(len(model_list)))],
               prop={'size': 8})

    plt.tight_layout()

    if random:
        plt.savefig('{0}/random_bias_vs_alpha.png'.format(plot_dir))
    else:
        plt.savefig('{0}/bias_vs_alpha.png'.format(plot_dir))

    plt.clf()


def plot_bias_vs_size(args, random=False):
    """Plots shape bias vs. stimulus size.

    :param args: command line arguments.
    :param random: true if plotting N random models + their average."""

    if random:
        model_list = ['{0}_{1}'.format(args.model, i) for i in range(1, args.N + 1)]
        model_list.insert(0, args.model)

        colors = ['#9c9c9c' for i in range(args.N)]
        colors.insert(0, '#16a1d9')

        labels = [str(i) for i in range(1, args.N + 1)]
        labels.insert(0, 'Average')

        plot_alphas = [0.4 for i in range(args.N)]
        plot_alphas.insert(0, (1))
    else:
        model_list = get_model_list()

        colors = ['#e274d0', '#e274d0', '#5ea6c6', '#5ea6c6',
                  '#57ad74', '#c69355', '#e6556c']
        labels = ['ResNet-50', 'ResNet-50 (Random)', 'ViT-B/16', 'ViT-B/16 (Random)',
                  'DINO ResNet-50', 'CLIP ViT-B/16', 'SAYCam-S']
        styles = ['solid', 'dashed', 'solid', 'dashed',
                  'solid', 'solid', 'solid']
        markers = ['o', 'o', '^', '^',
                   'o', '^', 'o']

    if args.novel:
        sim_dir = 'novel'
    else:
        sim_dir = 'geirhos'

    if args.classification:
        class_str = 'classifications/'
    else:
        class_str = ''

    if args.bg:
        bg_str = 'background_{0}/'.format(args.bg)
    else:
        bg_str = ''

    alignments = ['-unaligned', '-aligned']
    alignment_labels = ['Unaligned Shape', 'Aligned Shape']

    percent_ints = [20, 40, 60, 80, 100]

    if random:
        if args.bg:
            bg_str_temp = bg_str[:-1]
        else:
            bg_str_temp = bg_str

        plot_dir = 'figures/{0}/{1}{2}'.format(args.model, class_str, bg_str_temp)
    elif not args.all_models:  # For now, this only works for models in model_list
        idx = model_list.index(args.model)
        colors = [colors[idx]]
        labels = [labels[idx]]
        styles = [styles[idx]]
        markers = [markers[idx]]
        model_list = [model_list[idx]]

        if args.bg:
            bg_str_temp = bg_str[:-1]
        else:
            bg_str_temp = bg_str

        plot_dir = 'figures/{0}/{1}{2}'.format(args.model, class_str, bg_str_temp)
    else:
        if args.bg:
            bg_str_temp = bg_str[:-1]
        else:
            bg_str_temp = bg_str

        plot_dir = 'figures/{0}{1}'.format(class_str, bg_str_temp)

    model_dict = {key: {a: {"20": 0, "40": 0, "60": 0, "80": 0, "100": 0}
                        for a in alignments}
                  for key in model_list}

    for i in range(len(model_list)):
        model = model_list[i]

        for a in range(len(alignments)):
            alignment = alignments[a]

            for s in range(len(percent_ints)):
                size = percent_ints[s]

                prop_dir = 'results/{0}/{1}{2}{3}-alpha1-size{4}{5}/' \
                           'proportions_avg.csv'.format(model, class_str, bg_str,
                                                        sim_dir, size, alignment)

                props = pd.read_csv(prop_dir)
                shape_bias = props.at[0, 'Shape Match Closer']
                model_dict[model][alignment][str(size)] = shape_bias

    for a in range(len(alignments)):
        plt.clf()
        plt.axhline(0.5, color='#808080', linestyle=(0, (1, 3)))  # Chance line

        alignment = alignments[a]
        label = alignment_labels[a]

        for i in reversed(range(len(model_list))):
            if random:
                plt.plot(percent_ints, list(model_dict[model][alignment].values()),
                         label=labels[i], color=colors[i], marker=markers[i],
                         markersize=7.5, markeredgecolor='black', markeredgewidth=0.5,
                         markerfacecolor=colors[i], alpha=plot_alphas[i])
            else:
                plt.plot(percent_ints, list(model_dict[model][alignment].values()),
                         label=labels[i], color=colors[i], linestyle=styles[i],
                         marker=markers[i], markersize=7.5, markeredgecolor='black',
                         markeredgewidth=0.5, markerfacecolor=colors[i])

        plt.xticks(ticks=percent_ints)
        plt.title("{0} Stimuli, {1}".format(sim_dir.capitalize(), label))
        plt.xlabel('Percent of Original Size', fontsize=10)
        plt.ylabel('Proportion of Shape Decisions', fontsize=10)
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 1))

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend([handles[idx] for idx in reversed(range(len(model_list)))],
                   [labels[idx] for idx in range(len(model_list))],
                   prop={'size': 8})

        if random:
            plt.savefig('{0}/random_bias_vs_size_{1}_{2}.png'.format(plot_dir, sim_dir,
                                                                     alignment[1:]))
        else:
            plt.savefig('{0}/bias_vs_size_{1}_{2}.png'.format(plot_dir, sim_dir,
                                                              alignment[1:]))
