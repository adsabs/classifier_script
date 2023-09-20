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
                params={'q':'*:*', 'wt':'json', 'fq':'{!bitset}', 'fl':'bibcode,title,abstract,keyword,body', 'rows':len(bibs)},    
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

    # Test data from a list of bibcodes querying the API
    bibcodes = ['2017ApJ...845..161A', '2012ApJ...750..125A']
    url = config_dict['API_URL']
    token = config_dict['API_TOKEN']
    # mdata = get_records(bibcodes, token, url)

    # Previously generated and classified data
    df_full = pd.read_csv(config_dict['DATA_FULL_SAMPLE'])
    df_truth = pd.read_csv(config_dict['DATA_GROUND_TRUTH'])
    
    # 3 for intital test
    nn = 3
    test_bibs = list(df_full['bibcode'].values[:nn])
    test_abs = list(df_full['abstract'].values[:nn])
    # import pdb;pdb.set_trace()

    list_of_categories, list_of_scores = batch_assign_SciX_categories(list_of_texts=test_abs)
    # list_of_categories, list_of_scores = article_assign_SciX_categories(list_of_texts=test_abs)

    # Category Score Order
    ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']

    out_dict = {'bibcode': test_bibs,
                'category': list_of_categories,
                'score': list_of_scores}

    out_df = pd.DataFrame(out_dict)

    # Merge 




    import pdb;pdb.set_trace()
