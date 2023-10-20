WORKING_DIR = '~/Code/SciX_Classifier'
API_URL = "https://api.adsabs.harvard.edu/v1" # ADS API URL
API_TOKEN = ''
# Config file save directory
CONFIG_DIR = '/Users/thomasallen/Code/SciX_Classifier/config_files/'
# Input Data
DATA_FULL_SAMPLE = '/Users/thomasallen/Code/SciX_Classifier/data/full_sample.csv'
DATA_GROUND_TRUTH = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth.csv'
TEST_DATA_SOURCE = "Classified_CSV" # "Article', 'Classified_CSV'
DATA_SAMPLE_CLASSIFIED = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_sample_classified.csv' # Initial classified sample
# DATA_SAMPLE_CLASSIFIED_NEW = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_sample_classified_new.csv' # Latest classified sample
DATA_SAMPLE_CLASSIFIED_NEW = "/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_sample_classified_abstract.csv"
# Classification Parameters
RUN_SAMPLE_CLASSIFICATION = "no"
CLASSIFICATION_INPUT_TEXT = "abstract"
# CLASSIFICATION_INPUT_TEXT = 'Abstract' # 'Title', 'Abstract', 'Body'
# Classification Model
CLASSIFICATION_PRETRAINED_MODEL = "adsabs/ASTROBERT"
CLASSIFICATION_PRETRAINED_MODEL_REVISION = 'SciX-Categorizer'
# Plots
SHOW_BARCHART_COUNTS_ALL = False
MAKE_CATEGORY_BOXPLOTS = True
SHOW_CATEGORY_BOXPLOTS = False
BOXPLOT_SAVE_DIR = '/Users/thomasallen/Code/SciX_Classifier/figures/Score_Boxplots/'
