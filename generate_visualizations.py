import pandas as pd
import os
import json
from random import sample


def write_html(model_type, simulation):
    """This function creates an HTML page for a given model and simulation type (triplets
    or cartoon quadruplets). Each page contains visualizations of each triplet/quadruplet
    and the model's responses to the stimuli. The HTML script is written to a file.

    :param model_type: saycam, resnet50, etc.
    :param simulation: 'triplets' for Geirhos triplets, 'cartoon' for cartoon dataset."""

    im_titles = ['Anchor', 'Shape Match', 'Texture Match']

    if simulation == 'triplets':
        stimuli = json.load(open('geirhos_triplets.json'))['all']
        stim_dir = 'stimuli-shape/style-transfer/'
        result_dir = 'results/' + model_type + '/similarity/'
        dataset = 'Geirhos Style Transfer Dataset'
        term = 'Triplet'
        padding = '377px'
        size = 224
    else:
        stimuli = json.load(open('cartoon_trials.json'))['all']
        stim_dir = 'stimuli-shape/cartoon/'
        result_dir = 'results/' + model_type + '/cartoon/'
        dataset = 'Cartoon Dataset'
        term = 'Quadruplet'
        im_titles.append('Color Match')
        padding = '35px'
        size = 124

    stimuli = sample(stimuli, 20)

    html_string = ""

    # html start
    html_string += f"""
    <!doctype html>
    <html lang="en">
      <head>
        <!-- Required meta tags -->
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
        <link rel="stylesheet" href="bootstrap-custom.css">
        <link rel="stylesheet" href="style.css">
        <title>Model Response Visualizations: {model_type}</title>
      </head>
      <body>
        <div class="container">
          <br />
          <h2>Model Response Visualizations: {model_type}</h2>
          <h3>{dataset}</h3>
          <br></br>
    """

    for i in range(len(stimuli)):
        triplet = stimuli[i]  # [anchor, shape match, texture match, color match]
        results = pd.read_csv(result_dir + triplet[0][:-4] + '.csv')

        if simulation == 'triplets':
            result = results.loc[((results['Anchor'] == triplet[0][:-4])
                                  & (results['Shape Match'] == triplet[1][:-4])
                                  & (results['Texture Match'] == triplet[2][:-4]))]
        else:
            result = results.loc[((results['Anchor'] == triplet[0])
                                  & (results['Shape Match'] == triplet[1][:-4])
                                  & (results['Texture Match'] == triplet[2][:-4])
                                  & (results['Color Match'] == triplet[3][:-4]))]

        colors, match = assign_colors(result, simulation, im_titles)

        html_string += f"""
              <h5 style="padding-left:{padding}">{term} #{i + 1}</h5>
              <div class="row">
            """

        for j in range(len(triplet)):
            if simulation == 'triplets':
                shape_class = ''.join(k for k in triplet[j].split('-')[0] if not k.isdigit())
                im_path = stim_dir + shape_class + '/' + triplet[j]
            else:
                im_path = stim_dir + triplet[j]

            if im_titles[j] == match:
                html_string += f"""
                        <div class="col-{len(triplet)} text-center">
                          <strong>{im_titles[j]}</strong>
                          <figure class="figure">
                          <img src="../../{im_path}" class="figure-img img-fluid" style="border: 3px solid #10F310;width:{size}px;height:{size}px">
                          <figcaption class="figure-caption text-center"><i>{triplet[j][:-4]}</i></figcaption>
                          </figure>
                        """
            else:
                html_string += f"""
                        <div class="col-{len(triplet)} text-center">
                          <strong>{im_titles[j]}</strong>
                          <figure class="figure">
                          <img src="../../{im_path}" class="figure-img img-fluid" style="border: 3px solid #000000;width:{size}px;height:{size}px">
                          <figcaption class="figure-caption text-center"><i>{triplet[j][:-4]}</i></figcaption>
                          </figure>
                        """

            if j != 0:
                html_string += f"""
                    <table style="text-align:left">
                      <tr>
                        <td>cos:</td>
                        <td> </td>
                        <td style="color:{colors[im_titles[j]]['cos']}">{round(result.loc[((result['Metric'] == 'cos'))][im_titles[j].split(' ')[0] + ' Distance'].item(), 3)}</td>
                      </tr>
                      <tr>
                        <td>dot:</td>
                        <td> </td>
                        <td style="color:{colors[im_titles[j]]['dot']}">{round(result.loc[((result['Metric'] == 'dot'))][im_titles[j].split(' ')[0] + ' Distance'].item(), 3)}</td>
                      </tr>
                      <tr>
                        <td>ed:</td>
                        <td> </td>
                        <td style="color:{colors[im_titles[j]]['ed']}">{round(result.loc[((result['Metric'] == 'ed'))][im_titles[j].split(' ')[0] + ' Distance'].item(), 3)}</td>
                      </tr>
                    </table>
                    """

            html_string += f"""</div>\n"""

        html_string += f"""
          </div>
          <br></br>
        """

    # html end
    html_string += f"""
        </div>
        </div>
        </div>

        <!-- Optional JavaScript -->
        <!-- jQuery first, then Popper.js, then Bootstrap JS -->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js" integrity="sha384-9/reFTGAW83EW2RDu2S0VKaIzap3H56lZH81PoYlFhbGU+6BZp6G7niu735Sk7lN" crossorigin="anonymous"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" integrity="sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV" crossorigin="anonymous"></script>
      </body>
    </html>
        """

    with open(f'results/visualizations/{model_type}_{simulation}.html', 'w') as f:
        f.write(html_string)


def assign_colors(df, simulation, im_titles):
    """This function is a subroutine for write_html. It assigns the colors red/green to
    the text displaying the distances based on which one is considered closer.

    :param df: a DataFrame containing results for one triplet.
    :param simulation: 'triplets' for Geirhos, 'cartoon' for cartoon
    :param im_titles: list of titles for each image in a triplet.

    :return: a dictionary of color values and the closest image overall"""

    im_titles = [title for title in im_titles if title != 'Anchor']
    metrics = ['cos', 'dot', 'ed']
    colors = {key: {metric: None for metric in metrics} for key in im_titles}
    totals = {key: 0 for key in im_titles}

    for i in range(len(im_titles)):
        im_title = im_titles[i]

        for metric in metrics:
            if simulation == 'triplets':
                title_closer = df.loc[(df['Metric'] == metric)][im_title + ' Closer'].item()
            else:
                title_closer = df.loc[(df['Metric'] == metric)][im_title + ' Closer'].item()

            if title_closer == 1:
                colors[im_title][metric] = '#26A52D'
                totals[im_title] += 1
            else:
                colors[im_title][metric] = '#C41212'

    return colors, max(colors, key=lambda x: totals[x])


if __name__ == "__main__":

    try:
        os.mkdir('results/visualizations')
    except FileExistsError:
        pass

    model_list = ['saycam', 'saycamA', 'saycamS', 'saycamY', 'resnet50', 'clipRN50', 'clipRN50x4',
                  'clipRN50x16', 'clipViTB32', 'clipViTB16', 'dino_resnet50', 'alexnet', 'vgg16',
                  'swav', 'mocov2']

    for model in model_list:
        write_html(model, 'triplets')