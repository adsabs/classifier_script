#!bin/bash

# This script runs multiple experiments with different parameters

# Will Change parameters in the config.py file using sed

# Make backup of config file
cp config.py config.py.bak

# Run experiments

# Toggle Inference for ALL experiments

# NO Inference
sed -i '' -e 's/^RUN_SAMPLE_CLASSIFICATION = .*$/RUN_SAMPLE_CLASSIFICATION = "no"/' config.py

# YES Inference
# sed -i '' -e 's/^RUN_SAMPLE_CLASSIFICATION = .*$/RUN_SAMPLE_CLASSIFICATION = "yes"/' config.py

# Experiment 1
# SciX-Clategorizer version of adsabs/astroBERT
# Only Title as input Text

ECHO " "
ECHO "ONLY TITLE"

# Set MODEL Information
sed -i '' -e 's/^CLASSIFICATION_PRETRAINED_MODEL = .*$/CLASSIFICATION_PRETRAINED_MODEL = "adsabs\/ASTROBERT"/' config.py
sed -i '' -e 's/^CLASSIFICATION_PRETRAINED_MODEL_REVISIONg = .*$/CLASSIFICATION_INPUT_TEXT = "SciX-Categorizer"/' config.py

# Set INPUT Parameters
sed -i '' -e 's/^CLASSIFICATION_INPUT_TEXT = .*$/CLASSIFICATION_INPUT_TEXT = "title"/' config.py
sed -i '' -e 's/^DATA_SAMPLE_CLASSIFIED_NEW = .*$/DATA_SAMPLE_CLASSIFIED_NEW = "\/Users\/thomasallen\/Code\/SciX_Classifier\/data\/ground_truth_sample_classified_title.csv"/' config.py

# Turn off plots

sed -i '' -e 's/^SHOW_CATEGORY_BOXPLOTS = .*$/SHOW_CATEGORY_BOXPLOTS = False/' config.py

# Save config file
cp config.py config_files/config_only_title_model_v0.py

python run_experiments.py

# Experiment 2
# SciX-Clategorizer version of adsabs/astroBERT
# Only Title and Abstract as input Text

# Set MODEL Information
sed -i '' -e 's/^CLASSIFICATION_PRETRAINED_MODEL = .*$/CLASSIFICATION_PRETRAINED_MODEL = "adsabs\/ASTROBERT"/' config.py
sed -i '' -e 's/^CLASSIFICATION_PRETRAINED_MODEL_REVISIONg = .*$/CLASSIFICATION_INPUT_TEXT = "SciX-Categorizer"/' config.py

# Set INPUT Parameters
sed -i '' -e 's/^CLASSIFICATION_INPUT_TEXT = .*$/CLASSIFICATION_INPUT_TEXT = "abstract"/' config.py
sed -i '' -e 's/^DATA_SAMPLE_CLASSIFIED_NEW = .*$/DATA_SAMPLE_CLASSIFIED_NEW = "\/Users\/thomasallen\/Code\/SciX_Classifier\/data\/ground_truth_sample_classified_abstract.csv"/' config.py

# Turn off plots

sed -i '' -e 's/^SHOW_CATEGORY_BOXPLOTS = .*$/SHOW_CATEGORY_BOXPLOTS = False/' config.py

cp config.py config_files/config_only_title_abstract_model_v0.py

python run_experiments.py

