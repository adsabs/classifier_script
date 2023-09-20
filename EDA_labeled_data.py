import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from adsputils import setup_logging, load_config

# import pdb;pdb.set_trace()

config_dict = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__))))#, '.')


if __name__=="__main__":


    df = pd.read_csv(config_dict['DATA_GROUND_TRUTH'])

    import pdb;pdb.set_trace()

