import os
import sys
import json
import requests
from urllib.parse import urlencode, quote_plus
from time import perf_counter, time
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from batch_assign_SciX_categories import batch_assign_SciX_categories
# from astrobert.finetuning_2_seq_classification import batch_assign_SciX_categories
# from astrobert.finetuning_2_seq_classification import article_assign_SciX_categories
from adsputils import setup_logging, load_config

import pdb;pdb.set_trace()

config_dict = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__))))#, '.')
 
# def parse_inputs():
#     '''Parse and error check input for nearest_buildings function'''
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_file', required=True,\
#                         help='Path to config file')
#     args = parser.parse_args()
#     args_dict = vars(args)

    # Check inputs

    # Check Tables
    # try:
    #     config_dict = load_config(args_dict['config_file'])
    # except:
    #     print('Loading default config file')
    #     config_dict = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__))))#, '.')

    # return args_dict


# args_dict = parse_inputs()

# Now save the config dict to the following
def save_config_dict(config_dict):
    """Save config dict to json file"""

    config_dict['API_TOKEN'] = ''
    config_filename = f'config.{time()}.json'

    output_file = os.path.join(config_dict['CONFIG_DIR'], config_filename) 

    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=4)

save_config_dict(config_dict)
# import pdb;pdb.set_trace()

def classify_sample(return_df=False):
    """Classify a sample of text for comparison with ground truth"""

    # Previously generated and classified data
    df_full = pd.read_csv(config_dict['DATA_FULL_SAMPLE'])
    df_truth = pd.read_csv(config_dict['DATA_GROUND_TRUTH'])
    # remove the 'title' and 'abstract' columns from df_truth
    df_truth = df_truth.drop(columns=['title', 'abstract'])
    
    # import pdb;pdb.set_trace()
    df_in = df_truth.merge(df_full, on='bibcode', how='left')
    # 3 for intital test
    # nn = 3
    # test_bibs = list(df_in['bibcode'].values[:nn])
    # test_text = list(df_in['abstract'].values[:nn])
    # test_title = list(df_in['title'].values[:nn])
    test_bibs = list(df_in['bibcode'].values[:])
    test_abstract = list(df_in['abstract'].values[:])
    test_title = list(df_in['title'].values[:])
    # import pdb;pdb.set_trace()

    # Ordered list of Categories
    cat_list = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    

    # Initialize lists
    list_of_categories = []
    list_of_scores = []
    list_of_AST = []
    list_of_Helio = []
    list_of_Planet = []
    list_of_Earth = []
    list_of_Bio = []
    list_of_Phys = []
    list_of_Other = []
    list_of_Garbage = []

    # EnumeratedlLoop through each sample and assign categories
    # for bib, text in zip(test_bibs, test_text):
    for index, (bib, title, abstract) in enumerate(zip(test_bibs, test_title, test_abstract)):
        print()
        print("Text number: ", index)
        print(bib)
        print(f'Title: {title}')
        print(f'Abstract: {abstract}')
        # print(text)

        #check if abstract is nan
        if pd.isnull(abstract):
            abstract = ''

        # Check CLASSIFICATION_INPUT_TEXT for input text
        if config_dict['CLASSIFICATION_INPUT_TEXT'] == 'title':
            text = title
        elif config_dict['CLASSIFICATION_INPUT_TEXT'] == 'abstract':
            text = title + ' ' + str(abstract)

        print(f'Text: {text}')

        # import pdb;pdb.set_trace()

        # CLASSIFICATION_PRETRAINED_MODEL = 'adsabs/astroBERT'
        # CLASSIFICATION_PRETRAINED_MODEL_REVISION = 'SciX-Categorizer'
        # Assign categories
        tmp_categories, tmp_scores = batch_assign_SciX_categories(list_of_texts=[text],
                    pretrained_model_name_or_path=config_dict['CLASSIFICATION_PRETRAINED_MODEL'],
                    revision=config_dict['CLASSIFICATION_PRETRAINED_MODEL_REVISION'])

        # import pdb;pdb.set_trace()
        tmp_categories = tmp_categories[0]
        tmp_scores = tmp_scores[0]
        print(tmp_categories)
        print(tmp_scores)
        # import pdb;pdb.set_trace()

        # Append to lists
        list_of_categories.append(tmp_categories)
        list_of_scores.append(tmp_scores)
        list_of_AST.append(tmp_scores[0])
        list_of_Helio.append(tmp_scores[1])
        list_of_Planet.append(tmp_scores[2])
        list_of_Earth.append(tmp_scores[3])
        list_of_Bio.append(tmp_scores[4])
        list_of_Phys.append(tmp_scores[5])
        list_of_Other.append(tmp_scores[6])
        list_of_Garbage.append(tmp_scores[7])


    score_dict = {'bibcode': test_bibs,
                'category': list_of_categories,
                'score': list_of_scores,
                'new score AST': list_of_AST,
                'new score Helio': list_of_Helio,
                'new score Planetary': list_of_Planet,
                'new score Earth': list_of_Earth,
                'new score BPS': list_of_Bio,
                'new score Other PHY': list_of_Phys,
                'new score Other': list_of_Other,
                'new score Garbage': list_of_Garbage
                }

    score_df = pd.DataFrame(score_dict)

    
    # Join df_in and score_df on bibcode
    # import pdb;pdb.set_trace()
    df_out = df_in.merge(score_df, on='bibcode', how='left')
    df_out.to_csv(config_dict['DATA_SAMPLE_CLASSIFIED_NEW'], index=False)

    if return_df:
        return df_out


def relabel_categorical_categories(df, column='primaryClass'):
    """Rename categories in selected column of dataframe"""

    # import pdb;pdb.set_trace()
    mapping = {'Biology': 'BPS', 'FALSE': 'Other'}#, 'False': 'Other'}
    # mapping = {'Biology': 'BPS', 'FALSE': 'Other', 'False': 'Other'}

    # if there are NaNs in the column, replace them with 'FALSE'j                                       
    df[column] = df[column].fillna('False')
    # import pdb;pdb.set_trace()

    df[column] = df[column].astype('category')
    df[column] = df[column].cat.rename_categories(mapping)

    # import pdb;pdb.set_trace()

    return df

def plot_boxplot_category(df, cat, categories, short_categories, index,new_scores=True, column='primaryClass',show_plot=False):
    """Plot boxplot of scores for each category"""

    # categories = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    # short_categories = ['AST', 'Helio', 'Planetary', 'Earth', 'BPS', 'Other PHY', 'Other', 'Text Garbage']
    print()
    print("Beginning category: ", cat)
    # import pdb;pdb.set_trace()

    print(f"Score Data from {config_dict['DATA_SAMPLE_CLASSIFIED_NEW']}")

    # Use a list comprehension to create a list of the new score categories from the short categories
    new_score_categories = [f'new score {cat}' for cat in short_categories]
    # Now do the same for the score categories
    score_categories = [f'score {cat}' for cat in short_categories]
    # keep_categories = ['bibcode'] + score_categories + new_score_categories
    if new_scores:
        keep_categories = ['bibcode'] + new_score_categories
    else:
        keep_categories = ['bibcode'] + score_categories

    short_cat = short_categories[index] 
    # import pdb;pdb.set_trace()
    df_cat = df[df[column] == cat]
    df_cat = df_cat[keep_categories]

    # if there are no papers in the category, skip it
    if len(df_cat) == 0:
        print(f'No papers in category {cat}')
        return None

    # df_cat = df_cat[keep_categories]
    # import pdb;pdb.set_trace()
    # Transform the current dataframe into a long form, whith one column for the score, one column for the new score, and one column for the category
    # df_cat_long = pd.melt(df_cat, id_vars=['primaryClass'], value_vars=[f'score {short_cat}', f'new score {short_cat}'])
    df_cat_long = pd.melt(df_cat, id_vars=['bibcode'], value_vars=keep_categories)
    # stack the two dataframes
    # df_cat_long = pd.concat([df_cat_long1, df_cat_long2])
    # dfns = df_cat_long[df_cat_long['variable'] == f'new score {short_cat}']
    # import pdb;pdb.set_trace()

    pretrained_model = config_dict['CLASSIFICATION_PRETRAINED_MODEL']
    pretrained_model = pretrained_model.replace('/', '_')

    xs = 14
    ys = 8

    fig, ax = plt.subplots(figsize=(xs, ys))
    plot_box_score = sns.boxplot(x='variable', y='value', data=df_cat_long,ax=ax)
    plot_box_score.set(title=f"Boxplot of scores for articles classified as {cat}\n{config_dict['CLASSIFICATION_INPUT_TEXT']}\n{config_dict['CLASSIFICATION_PRETRAINED_MODEL']} - {config_dict['CLASSIFICATION_PRETRAINED_MODEL_REVISION']}\n{column}")
    plot_filepath = f'{config_dict["BOXPLOT_SAVE_DIR"]}boxplot_scores_{cat}_{column}_{config_dict["CLASSIFICATION_INPUT_TEXT"]}_{pretrained_model}_{config_dict["CLASSIFICATION_PRETRAINED_MODEL_REVISION"]}.png'
    # import pdb;pdb.set_trace()
    plt.savefig(plot_filepath)
    if show_plot:
        plt.show()
    plt.close()
    plt.clf()

# CLASSIFICATION_INPUT_TEXT = 'abstract' # 'title', 'abstract', 'body'
# CLASSIFICATION_PRETRAINED_MODEL = 'adsabs/astroBERT'
# CLASSIFICATION_PRETRAINED_MODEL_REVISION = 'SciX-Categorizer'
# Plots
    # fig, ax = plt.subplots(figsize=(xs, ys))
    # plot_box_newscore = sns.boxplot(x='variable', y='value', data=df_cat_long2,ax=ax)
    # plot_box_newscore.set(title=f'Boxplot of new scores for articles classified as {short_cat}')
    # plt.show()
    # plt.close()
    # plt.clf()

    print("Finished with category: ", cat)


if __name__ == "__main__":

    #List of categories
    categories = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    short_categories = ['AST', 'Helio', 'Planetary', 'Earth', 'BPS', 'Other PHY', 'Other', 'Text Garbage']
    # score_categories = ['score AST', 'score Helio', 'score Planet', 'score Earth', 'score Bio', 'score Phys', 'score Other', 'score Garbage']
    # new_score_categories = ['new score AST', 'new score Helio', 'new score Planet', 'new score Earth', 'new score Bio', 'new score Phys', 'new score Other', 'new score Garbage']

    # Use a list comprehension to create a list of the new score categories from the short categories
    new_score_categories = [f'new score {cat}' for cat in short_categories]
    # Now do the same for the score categories
    score_categories = [f'score {cat}' for cat in short_categories]
    keep_categories = ['bibcode'] + score_categories + new_score_categories
    keep_categories1 = ['bibcode'] + score_categories
    keep_categories2 = ['bibcode'] + new_score_categories

    # Run sample classification or
    # Load previously generated and classified data
    if config_dict['RUN_SAMPLE_CLASSIFICATION'] == 'yes':
        # start the perf ounter
        t0 = perf_counter()
        # Run sample classification
        # Check config file for settings
        df = classify_sample(return_df=True)
        # stop the perf counter
        t1 = perf_counter()
        # calculate the elapsed time
        elapsed_time = t1 - t0
        print(f'Elapsed time: {elapsed_time} seconds')
    elif config_dict['RUN_SAMPLE_CLASSIFICATION'] == 'no':
        # Do not run sample classification
        # df = pd.read_csv(config_dict['DATA_SAMPLE_CLASSIFIED'])
        df = pd.read_csv(config_dict['DATA_SAMPLE_CLASSIFIED_NEW'])


    # import pdb;pdb.set_trace()
    # rename categories in df column 'primaryClass'
    df = relabel_categorical_categories(df, column='primaryClass')
    try:
        df = relabel_categorical_categories(df, column='secondaryClass')
    except:
        print('Secondary class not present in dataframe')

    # Now rename the new_score_Garbage column to new_score_Text_Garbage
    df = df.rename(columns={'new score Garbage': 'new score Text Garbage'})
    # Now rename Score Text Garbage to score Text Garbage
    df = df.rename(columns={'Score Text Garbage': 'score Text Garbage'})


    # Variables of interest
    # primaryClass secondareyClass score... new score...

    # Create summary table that shows the number of papers in each category of primaryClass
    df_summary_primary_class = df.groupby('primaryClass').size().reset_index(name='counts')
    df_summary_secondary_class = df.groupby('secondaryClass').size().reset_index(name='counts')

    print(df_summary_primary_class)
    print(df_summary_secondary_class)
    # df_summary_classes = df_summary_classes[['primaryClass']]

    ############################
    # Plotting
    ############################

    # First lest plot the number of papers in each category
    if config_dict['SHOW_BARCHART_COUNTS_ALL']:
        sns.barplot(x='primaryClass', y='counts', data=df_summary_primary_class)
        plt.show()
        plt.close()
        plt.clf()

    # import pdb;pdb.set_trace()
    # Now lets loop through each category and create a boxplot of the scores
    if config_dict['MAKE_CATEGORY_BOXPLOTS']:

        for index, cat in enumerate(categories):

            plot_boxplot_category(df, cat, categories, short_categories, index,new_scores=True, column='primaryClass',show_plot=config_dict['SHOW_CATEGORY_BOXPLOTS'])
            plot_boxplot_category(df, cat, categories, short_categories, index,new_scores=True, column='secondaryClass',show_plot=config_dict['SHOW_CATEGORY_BOXPLOTS'])
            # plot_boxplot_category(df, cat, short_categories, keep_categories, index, column='primaryClass')
        
 
    # import pdb;pdb.set_trace()
