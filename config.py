WORKING_DIR = '~/Code/SciX_Classifier'
API_URL = "https://api.adsabs.harvard.edu/v1" # ADS API URL
API_TOKEN = ''
# Config file save directory
CONFIG_DIR = '/Users/thomasallen/Code/SciX_Classifier/config_files/'
# Input Data
DATA_FULL_SAMPLE = '/Users/thomasallen/Code/SciX_Classifier/data/full_sample.csv'
DATA_GROUND_TRUTH = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth.csv'
DATA_GROUND_TRUTH_ALL = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_all_curated.csv'
# DATA_GROUND_TRUTH_ALL = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_all.csv'
DATA_GROUND_TRUTH_ALL_PICKLE = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_all.pkl'
DATA_GROUND_TRUTH_ALL_JSON = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_all.json'
TEST_DATA_SOURCE = "Classified_CSV" # "Article', 'Classified_CSV'
DATA_SAMPLE_CLASSIFIED = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_sample_classified.csv' # Initial classified sample
DATA_EXTRA_HELIO = '/Users/thomasallen/Code/SciX_Classifier/data/helio_nature_science_bibcode_list.csv'
DATA_EXTRA_PLANETARY = '/Users/thomasallen/Code/SciX_Classifier/data/ps_nature_science_bibcode_list.csv'
# DATA_SAMPLE_CLASSIFIED_NEW = '/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_sample_classified_new.csv' # Latest classified sample
DATA_SAMPLE_CLASSIFIED_NEW = "/Users/thomasallen/Code/SciX_Classifier/data/ground_truth_sample_classified_title_abstract_no_labels_chkp32100.csv"
# Classification Parameters
RUN_SAMPLE_CLASSIFICATION = "no"
CLASSIFICATION_INPUT_TEXT = "title abstract"
# CLASSIFICATION_INPUT_TEXT = 'Abstract' # 'title', 'abstract' 'title abstract'
# Classification Model
PUBLISHED_MODEL = False
CLASSIFICATION_PRETRAINED_MODEL = "adsabs/ASTROBERT"
# CLASSIFICATION_PRETRAINED_MODEL_UNPUBLISHED = "/Users/thomasallen/Code/SciX_Classifier/models/checkpoint-32100"
CLASSIFICATION_PRETRAINED_MODEL_UNPUBLISHED = "/Users/thomasallen/Code/SciX_Classifier/models/checkpoint-32100"
CLASSIFICATION_PRETRAINED_MODEL_REVISION = "SciX-Categorizer"
CLASSIFICATION_INPUT_TEXT = "title abstract"
# Plots
SHOW_BARCHART_COUNTS_ALL = False
MAKE_CATEGORY_BOXPLOTS = True
SHOW_CATEGORY_BOXPLOTS = False
BOXPLOT_SAVE_DIR = '/Users/thomasallen/Code/SciX_Classifier/figures/Score_Boxplots/'
EXAMINE_CATAGORIES = False

TEST_THRESHOLDS = True
TEST_THRESHOLDS_METHOD = "max"
TEST_LABELS = False

EXPLORE_EARTH_SCIENCE = True
EARTH_SCIENCE_THRESHOLD = 0.015
