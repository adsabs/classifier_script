import time
import os
import json

import requests
from adsputils import setup_logging, load_config
# from tqdm import tqdm
# import ptree

#from ..utilities import remove_control_chars
# import remove_control_chars as rcc

# load API config
#solr_config = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))
solr_config = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__), '.')))


def harvest_solr(bibcodes_list, start_index=0, fields='bibcode, title, abstract'):
    ''' Harvests citations for an input list of bibcodes using the ADS API.

    It will perform minor cleaning of utf-8 control characters.
    Log in output_dir/logs/harvest_clean.log -> tail -f logs/harvest_clean.log .
    bibcodes_list: a list of bibcodes to harvest citations for.
    paths_list:_list: a list of paths to save the output
    start_index (optional): starting index for harvesting
    fields (optional): fields to harvest from the API. Default is 'bibcode, title, abstract'.
    '''

    # save the log next to the bibcode files
    logger = setup_logging('harvest_clean', proj_home=os.path.dirname('harvest_log.txt'))

    # COnvert list of bibcodes to a set for comparison later

    out_path = 'data/'

    # starting index of harvesting
    idx=start_index

    # params for stats
    # number of bibcodes to harvest at once
    step_size = 2000
    # step_size = 20

    # json dict to dump
    dataset = {}

    logger.info('Start of harvest')
    #start progress bar
    # pbar = tqdm(total=len(bibcodes_list), initial=start_index)
    # print('checkpoint harvest_solr.py')
    # import pdb;pdb.set_trace()

    # loop through list of bibcodes and query solr
    while idx<len(bibcodes_list):

        start_time = time.perf_counter()
        # string to log
        to_log = ''

        # limit attempts to 10
        attempts = 0
        successful_req = False

        # extract next step_size list
        input_bibcodes = bibcodes_list[idx:idx+step_size]
        bibcodes = 'bibcode\n' + '\n'.join(input_bibcodes)

        # start attempts
        while (not successful_req) and (attempts<10):
            r_json = None
            r = requests.post(solr_config['API_URL']+'/search/bigquery',
                    params={'q':'*:*', 'wt':'json', 'fq':'{!bitset}', 'fl':fields, 'rows':len(input_bibcodes)},
                              headers={'Authorization': 'Bearer ' + solr_config['API_TOKEN'], "Content-Type": "big-query/csv"},
                              data=bibcodes)

            # check that request worked
            # proceed if r.status_code == 200
            # if fails, log r.text, then repeat for x tries
            if r.status_code==200:
                successful_req=True
            else:
                to_log += 'REQUEST {} FAILED: CODE {}\n'.format(attempts, r.status_code)
                to_log += str(r.text)+'\n'

            # inc count
            attempts+=1

        # after request
        if successful_req:
            #extract json
            r_json = r.json()

            # add to stat counts
            # astronomy_count += len(r_json['response']['docs'])

            # info to log
            to_log += 'Harvested links up to {}\n'.format(idx)
            # to_log += 'Running astronomy count={}, body count={}, ack count={}\n'.format(astronomy_count,body_count,ack_count)


        # if not successful_req
        else:
            # add to log
            to_log += 'FAILING BIBCODES: {}\n'.format(input_bibcodes)

#             # raise error
#             r.raise_for_status()

        print()
        print('index')
        print(idx)
        # import pdb;pdb.set_trace()

        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print()
        print()
        print('index')
        print(idx)
        print()
        print('Time for loop segment')
        print(total_time)

        # with open('data/abstracts/test.txt','w') as f:
        #     f.write("test")

        # import pdb;pdb.set_trace()

        # pause to not go over API rate limit
        if len(bibcodes_list)>step_size:
            time.sleep(45)

        # ALWAYS DO:
        # increment counter for next batch
        idx+=step_size
        # update progress bar
        # pbar.update(step_size)
        #update log
        logger.info(to_log)

    # print('checkpoint harvest_solr.py')
    # import pdb;pdb.set_trace()

    return transform_r_json(r_json)


def transform_r_json(r_json):
    """
    Extract the needed information from the json response from the solr query.
    """

    # extract the needed information
    # Bibcoded and titles are always present
    bibcodes = [doc['bibcode'] for doc in r_json['response']['docs']]
    titles = [doc['title'][0] for doc in r_json['response']['docs']] # without [0] it returns a list
    # abstracts = [doc['abstract'] for doc in r_json['response']['docs']]
    # Abstracts are not always present
    abstracts = []
    for doc in r_json['response']['docs']:
        if 'abstract' in doc:
            abstracts.append(doc['abstract'][0])
        else:
            abstracts.append('')

    # list of dictionaries with the bibcode, title, and abstract for each record
    record_list = [{'bibcode': bibcodes[i],
                    'title' : titles[i],
                    'abstract' : abstracts[i],
                    'text': f'{titles[i]} {abstracts[i]}'} for i in range(len(bibcodes))]

    # return bibcodes, titles, abstracts
    return record_list
