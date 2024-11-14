#!/usr/bin/env python
"""
"""

# __author__ = 'rca'
# __maintainer__ = 'rca'
# __copyright__ = 'Copyright 2015'
# __version__ = '1.0'
# __email__ = 'ads@cfa.harvard.edu'
# __status__ = 'Production'
# __credit__ = ['J. Elliott']
# __license__ = 'MIT'

import os
import csv
# import sys
# import time
# import json
import argparse
# import logging
# import traceback
# import warnings
# from urllib3 import exceptions
# warnings.simplefilter('ignore', exceptions.InsecurePlatformWarning)

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from adsputils import get_date
# from adsmsg import OrcidClaims
from ClassifierPipeline import classifier, tasks
# from ADSOrcid import updater, tasks
# from ADSOrcid.models import ClaimsLog, KeyValue, Records, AuthorInfo

# # ============================= INITIALIZATION ==================================== #

from adsputils import setup_logging, load_config
proj_home = os.path.realpath(os.path.dirname(__file__))
# global config
config = load_config(proj_home=proj_home)
logger = setup_logging('run.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', False))

# app = tasks.app

# =============================== FUNCTIONS ======================================= #

def load_model_and_tokenizer(pretrained_model_name_or_path=None, revision=None, tokenizer_model_name_or_path=None):
    """
    Load the model and tokenizer for the classification task, as well as the
    label mappings. Returns the model, tokenizer, and label mappings as a
    dictionary.

    Parameters
    ----------
    pretrained_model_name_or_path : str (optional) (default=None) Specifies the
        model name or path to the model to load. If None, then reads from the 
        config file.
    revision : str (optional) (default=None) Specifies the revision of the model
    tokenizer_model_name_or_path : str (optional) (default=None) Specifies the
        model name or path to the tokenizer to load. If None, then defaults to
        the pretrained_model_name_or_path.
    """
    # Define labels and ID mappings
    labels = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    id2label = {i:c for i,c in enumerate(labels) }
    label2id = {v:k for k,v in id2label.items()}

    # Define model and tokenizer
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL']
    if revision is None:
        revision = config['CLASSIFICATION_PRETRAINED_MODEL_REVISION']
    if tokenizer_model_name_or_path is None:
        tokenizer_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_model_name_or_path,
                                              revision=revision,
                                              do_lower_case=False)

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                    revision=revision,
                                    num_labels=len(labels),
                                    problem_type='multi_label_classification',
                                    id2label=id2label,
                                    label2id=label2id
                                        )
    # Output as dictionary
    model_dict = {'model': model,
                  'tokenizer': tokenizer,
                  'labels': labels,
                  'id2label': id2label,
                  'label2id': label2id}

    return model_dict

def prepare_records(records_path):
    """
    Takes a path to a .csv file of records and converts each record into a
    dictionary with the following keys: bibcode and text (a combination of 
    title and abstract). Sends each record to the classification queue.

    Parameters
    ----------
    records_path : str (required) (default=None) Path to a .csv file of records

    Returns
    -------
    no return
    """
    # Open .csv file and read in records
    # Note: that this method requres the input file to have the following
    # columns: bibcode, title, abstract
    with open(records_path, 'r') as f: 
        csv_reader = csv.reader(f)
        headers = next(csv_reader)

        for row in csv_reader:
            record = {}
            record['bibcode'] = row[0]
            record['text'] = row[1] + ' ' + row[2]

            print('testing message')
            # Now send record to classification queue
            # For Testing
            tasks.task_send_input_record_to_classifier(record)
            # For Production
            # tasks.task_send_input_record_to_classifier.delay(record)


def score_record(record):
    """
    Provide classification scores for a record using the following
        categories:
            0 - Astronomy
            1 - HelioPhysics
            2 - Planetary Science
            3 - Earth Science
            4 - Biological and Physical Sciences
            5 - Other Physics
            6 - Other
            7 - Garbage

    Parameters
    ----------
    records_path : str (required) (default=None) Path to a .csv file of records

    Returns
    -------
    records : dictionary with the following keys: bibcode, text,
                categories, scores, and model information
    """
    # Load model and tokenizer
    model_dict = load_model_and_tokenizer()

    text = f"{record['title']} {record['abstract']}"

    # Classify record
    record['categories'], record['scores'] = classifier.batch_assign_SciX_categories(
                                [text],model_dict['tokenizer'],
                                model_dict['model'],model_dict['labels'],
                                model_dict['id2label'],model_dict['label2id'])

    # Because the classifier returns a list of lists so it can batch process
    # Take only the first element of each list
    record['categories'] = record['categories'][0]
    record['scores'] = record['scores'][0]

    # Append model information to record
    # record['model'] = model_dict['model']
    record['model'] = model_dict


    # print("Record: {}".format(record['bibcode']))
    # print("Text: {}".format(record['text']))
    # print("Categories: {}".format(record['categories']))
    # print("Scores: {}".format(record['scores']))

    return record

def classify_record_from_scores(record):
    """
    Classify a record after it has been scored. 

    Parameters
    ----------
    record : dictionary (required) (default=None) Dictionary with the following
        keys: bibcode, text, categories, scores, and model information

    Returns
    -------
    record : dictionary with the following keys: bibcode, text, categories,
        scores, model information, and Collections
    """

    # Fetch thresholds from config file
    thresholds = config['CLASSIFICATION_THRESHOLDS']
    # print('Thresholds: {}'.format(thresholds))


    scores = record['scores']
    categories = record['categories']
    # max_score_index = scores.index(max(scores))
    # max_category = categories[max_score_index]
    # max_score = scores[max_score_index]

    meet_threshold = [score > threshold for score, threshold in zip(scores, thresholds)]

    # Extra step to check for "Earth Science" articles miscategorized as "Other"
    # This is expected to be less neccessary with improved training data
    if config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] is True:
        # print('Additional Earth Science Processing')
        # import pdb;pdb.set_trace()
        if meet_threshold[categories.index('Other')] is True:
            # If Earth Science score above additional threshold
            if scores[categories.index('Earth Science')] \
                    > config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD']:
                meet_threshold[categories.index('Other')] = False
                meet_threshold[categories.index('Earth Science')] = True

    # Append collections to record
    record['collections'] = [category for category, threshold in zip(categories, meet_threshold) if threshold is True]
    record['earth_science_adjustment'] = config['ADDITIONAL_EARTH_SCIENCE_PROCESSING']

    return record


def index_record():
    """
    Indexes a record
    """
    pass


# =============================== MAIN ======================================= #

# To test the classifier
# python run.py -n -r ClassifierPipeline/tests/stub_data/stub_new_records.csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process user input.')

    parser.add_argument('-n',
                        '--new_records',
                        dest='new_records',
                        action='store_true',
                        help='Process new records')

    parser.add_argument('-v',
                        '--validate',
                        dest='validate',
                        action='store_true',
                        help='Return list to manually validate classifications')

    parser.add_argument('-r',
                        '--records',
                        dest='records',
                        action='store',
                        help='Path to comma delimited list of new records' +
                             'to process: columns: bibcode, title, abstract')


    args = parser.parse_args()


    if args.records:
        records_path = args.records
        print(records_path)
        # Open .csv file and read in records
        # Convert records to send to classifier

    # import pdb;pdb.set_trace()
    if args.validate:
        print("Validating records")

    # import pdb;pdb.set_trace()
    if args.new_records:
        print("Processing new records")
        prepare_records(records_path)
        # records = score_records(records_path)

        # for record in records:
            # print("Record: {}".format(record['bibcode']))
            # print("Text: {}".format(record['text']))
            # print("Categories: {}".format(record['categories']))
            # print("Scores: {}".format(record['scores']))
        # records = classify_records_from_scores(records)

    print("Done")
    import pdb;pdb.set_trace()
