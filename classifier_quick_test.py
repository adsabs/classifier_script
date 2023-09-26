import os
import sys
import json
import requests
from urllib.parse import urlencode, quote_plus

import pandas as pd

from astrobert.finetuning_2_seq_classification import batch_assign_SciX_categories
# from astrobert.finetuning_2_seq_classification import article_assign_SciX_categories
from adsputils import setup_logging, load_config

# import pdb;pdb.set_trace()

config_dict = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__))))#, '.')


# import pdb;pdb.set_trace()

def get_records(bibs, token, url):
    # encoded_query = urlencode({"q": "*:*",
    #                        "fl": "bibcode,title,abstract,keyword,body",
    #                        "rows": len(bibs),
    #                       })
    # payload = "\n".join(bibs)
    # results = requests.post("https://api.adsabs.harvard.edu/v1/search/bigquery?{}".format(encoded_query), \
    #                    headers={'Authorization': 'Bearer ' + token}, \
    #                    data=payload)

    # # adata = results.json()['response']['docs']
    # adata = results.json()
    # return adata

    # limit attempts to 10
    attempts = 0
    successful_req = False
    # extract next step_size list
    bibcodes = 'bibcode\n' + '\n'.join(bibs)
    # start attempts                                                                                                                                                                                                                           
    while (not successful_req) and (attempts<10):                                                                                                                                                                                              
        r_json = None                                                                                                                                                                                                                          
        r = requests.post(url+'/search/bigquery',            
                params={'q':'*:*', 'wt':'json', 'fq':'{!bitset}', 'fl':'bibcode,title,abstract,keyword,body,ack', 'rows':len(bibs)},    
                          headers={'Authorization': 'Bearer ' + token, "Content-Type": "big-query/csv"},    
                          data=bibcodes)                                        
                                                                                
        # check that request worked                                             
        # proceed if r.status_code == 200                                       
        # if fails, log r.text, then repeat for x tries                         
        if r.status_code==200:                                                  
            successful_req=True                                                 
        else:                                                                   
            print('Not Successful')
                                                                                
        # inc count                                                             
        attempts+=1                                                             
                                                                            
    # after request                                                         
    if successful_req:                                                      
        return r.json()
    
    # if not successful_req                                                 
    else:                                                                   
        return []
                                                                            

if __name__ == "__main__":

    # Select test data source
    if config_dict['TEST_DATA_SOURCE'] == 'Article':
        # Test data from a list of bibcodes querying the API
        bibcodes = ['2017ApJ...845..161A', '2012ApJ...750..125A']
        # bibcodes = ['2017ApJ...845..161A']#;, '2012ApJ...750..125A']
        url = config_dict['API_URL']
        token = config_dict['API_TOKEN']
        mdata = get_records(bibcodes, token, url)

        
        articles = mdata['response']['docs']

        # Convert list of dictionaries to two lists of bibcodes and body of text
        test_bibs = [article['bibcode'] for article in articles]
        # test_text = [article['abstract'] for article in articles]
        test_text = [article['body'] for article in articles]

    elif config_dict['TEST_DATA_SOURCE'] == 'Classified_CSV':
        # import pdb;pdb.set_trace()

        # Previously generated and classified data
        df_full = pd.read_csv(config_dict['DATA_FULL_SAMPLE'])
        df_truth = pd.read_csv(config_dict['DATA_GROUND_TRUTH'])
        
        # import pdb;pdb.set_trace()
        df_in = df_truth
        # 3 for intital test
        nn = 3
        test_bibs = list(df_in['bibcode'].values[:nn])
        test_text = list(df_in['abstract'].values[:nn])
        # import pdb;pdb.set_trace()

    # import pdb;pdb.set_trace()
    # list_of_categories, list_of_scores = batch_assign_SciX_categories(list_of_texts=test_text)
    # list_of_categories, list_of_scores = article_assign_SciX_categories(list_of_texts=test_abs)

    # loop through each sample and assign categories

    list_of_categories = []
    list_of_scores = []

    for bib, text in zip(test_bibs, test_text):
        print(bib)
        print(text)
        tmp_categories, tmp_scores = batch_assign_SciX_categories(list_of_texts=[text])
        tmp_categories = tmp_categories[0]
        tmp_scores = tmp_scores[0]
        print(tmp_categories)
        print(tmp_scores)
        list_of_categories.append(tmp_categories)
        list_of_scores.append(tmp_scores)

    score_dict = {'bibcode': test_bibs,
                'category': list_of_categories,
                'score': list_of_scores}

    score_df = pd.DataFrame(score_dict)

    
    # Join df_in and score_df on bibcode
    # import pdb;pdb.set_trace()
    df_out = df_in.merge(score_df, on='bibcode', how='left')

    import pdb;pdb.set_trace()
    # Category Score Order
    ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']



    import pdb;pdb.set_trace()
