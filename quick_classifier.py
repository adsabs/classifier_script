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

# import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from adsputils import get_date
# from adsmsg import OrcidClaims
from ClassifierPipeline import classifier, tasks
# from ADSOrcid import updater, tasks
# from ADSOrcid.models import ClaimsLog, KeyValue, Records, AuthorInfo
from run import score_record, classify_record_from_scores, prepare_records
from harvest_solr import harvest_solr

# # ============================= INITIALIZATION ==================================== #

from adsputils import setup_logging, load_config
proj_home = os.path.realpath(os.path.dirname(__file__))
global config
config = load_config(proj_home=proj_home)
logger = setup_logging('run.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', False))

# app = tasks.app

# =============================== FUNCTIONS ======================================= #

def write_batch_to_csv(batch, header, filename, mode='w', include_header=True):
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        if include_header:
            writer.writerow(header)
        writer.writerows(batch)

# =============================== MAIN ======================================= #

# python3 quick_classifier.py -r /proj/ads_abstracts/adsnlp/stub_bibcodes.csv
# python3 quick_classifier.py -r /proj/ads_abstracts/adsnlp/2023Sci.csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process user input.')


    parser.add_argument('-r',
                        '--records',
                        dest='records',
                        action='store',
                        help='Path to comma delimited list of new records' +
                             'to process: columns: bibcode, title, abstract')

    parser.add_argument('-s',
                        '--sort',
                        dest='sort',
                        action='store_true',
                        help='Sorts return list in ascending order according to' +
                             'the minimum collection score. Default is False.')

    args = parser.parse_args()


    if args.records:
        records_path = args.records
        out_path = records_path.replace('.csv', '_classified.csv')
        print(f'Reading in {records_path} may take a minute for large input files.')
        print(f'Will write output to {out_path}.')
    else:
        print("Please provide a path to a .csv file with records to process.")
        exit()
        # Open .csv file and read in records
        # Convert records to send to classifier

    # records = pd.read_csv(records_path)
    output_list = []
    output_batch = 500
    with open(records_path, 'r') as f:
        bibcodes = f.read().splitlines()

    # Start with just bibcodes
    # bibcodes = records['bibcode'].tolist()
    # If first line is 'bibcode' remove it
    if bibcodes[0]=='bibcode':
        bibcodes = bibcodes[1:]

    # Harvest Title and Abstract from Solr
    records = harvest_solr(bibcodes, start_index=0, fields='bibcode, title, abstract')

    # import pdb;pdb.set_trace()
    # Prepare records for classification
    # Check if just bibcodes or full records
    # if just bibcodes then get records from solr

    # import pdb;pdb.set_trace()
    # Initialize output 
    output_list = []
    output_batch = 500
    batch_count = 0

    header = 'bibcode,title,abstract,categories,scores,collections,collection_scores,earth_science_adjustment'
    
    # with open(out_path, 'w') as f:
    #     f.write('bibcode,title,abstract,categories,scores,collections,collection_scores,minimum_collection_score,earth_science_adjustment\n')
    

    # import pdb;pdb.set_trace()
    # Loop through records and classify
    # for index, record in enumerate(records):
    for index, record in enumerate(records):
        # record = harvest_solr([bibcode], fields='bibcode, title, abstract')
        # import pdb;pdb.set_trace()
        record = score_record(record)
        record = classify_record_from_scores(record)
        del record['model']
        del record['text']

        # Get scores for elements in collections
        collection_scores = []
        for collection in record['collections']:

                # meet_threshold[categories.index('Earth Science')] = True
            collection_scores.append(record['scores'][record['categories'].index(collection)]) 

        # import pdb;pdb.set_trace()
        # Sort collection scores and collections from highest to lowest
        collection_scores, record['collections'] = zip(*sorted(zip(collection_scores, record['collections']), reverse=True))

        # import pdb;pdb.set_trace()
        # Update records
        record['collection_scores'] = collection_scores
        # record['minimum_collection_score'] = min(collection_scores)
        # Remove "model" and "text keys from record

        # Overwrite record with updated record
        records[index] = record

        record_output = [record['bibcode'],record['title'],record['abstract'],', '.join(record['categories']),', '.join(map(str, record['scores'])),', '.join(record['collections']),', '.join(map(str, record['collection_scores'])),str(record['earth_science_adjustment'])]

        # import pdb;pdb.set_trace()
        output_list.append(record_output)
        # output_list.append([record['bibcode'],str(record['title']),[record['abstract']],record['categories'],record['scores'],record['collections'],record['collection_scores'],record['earth_science_adjustment']])

        # Write to file
        if ((index+1) % output_batch == 0) or (len(records) < 500 and index == len(records)-1):
            # with open(out_path, 'a') as f:
            #     for record in output_list:
            #         f.write(f"{record['bibcode']} {record['title']} {record['abstract']} {record['categories']} {record['scores']} {record['collections']} {record['collection_scores']} {record['earth_science_adjustment']}\n")
		    	
            if batch_count == 0:
                include_header = True
                mode = 'w'
            else:
                include_header = False
                mode = 'a'
            write_batch_to_csv(output_list, header.split(','), out_path, mode=mode, include_header=include_header)
            batch_count += 1
            output_list = []
            print(f"Processed {index+1} records")

        
    # Convert records to a dataframe
    # records = pd.DataFrame(records)

    # Sort if needed
    # if args.sort is True:
    #     records = records.sort_values(by=['minimum_collection_score'], ascending=True)

    # records.to_csv(out_path, index=False)



    print("Done")
    print(f"Results save to {out_path}")
