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

# import os
# import sys
# import time
# import json
# import argparse
# import logging
# import traceback
# import warnings
# from urllib3 import exceptions
# warnings.simplefilter('ignore', exceptions.InsecurePlatformWarning)

# from adsputils import get_date
# from adsmsg import OrcidClaims
# from ADSOrcid import updater, tasks
# from ADSOrcid.models import ClaimsLog, KeyValue, Records, AuthorInfo

# # ============================= INITIALIZATION ==================================== #

# from adsputils import setup_logging, load_config
# proj_home = os.path.realpath(os.path.dirname(__file__))
# config = load_config(proj_home=proj_home)
# logger = setup_logging('run.py', proj_home=proj_home,
#                         level=config.get('LOGGING_LEVEL', 'INFO'),
#                         attach_stdout=config.get('LOG_STDOUT', False))

# app = tasks.app

# =============================== FUNCTIONS ======================================= #




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process user input.')

    parser.add_argument('-n',
                        '--new_records',
                        dest='new records',
                        action='store_true',
                        help='Process new records')


    args = parser.parse_args()
