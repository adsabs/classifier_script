# from __future__ import absolute_import, unicode_literals
import adsputils
# from adsmsg import OrcidClaims
# from SciXClassifier import app as app_module
# from SciXClassifier import updater
# from SciXClassifier.exceptions import ProcessingException, IgnorableException
# from SciXClassifier.models import KeyValue
# from kombu import Queue
# import datetime
# import os

# ============================= INITIALIZATION ==================================== #

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
app = app_module.SciXClassifierCelery(
    "scixclassifier-pipeline",
    proj_home=proj_home,
    local_config=globals().get("local_config", {}),
)
app.conf.CELERY_QUEUES = (
    Queue("unclassified-queue", app.exchange, routing_key="unclassified-queue"),
)
logger = app.logger


# ============================= TASKS ============================================= #

# From Curators Daily Operations 

# Send data to the Classifier

# Populate database wit new data

# Return sorted classifications to Curators

# Query SOLR
#   - Finding records with given set of parameters (e.g. classification, model, etc.)


@app.task(queue="unclassified-queue")
def task_send_input_record_to_classifier(message):
    """
    Send a new record to the classifier


    :param message: contains the message inside the packet
        {
         'bibcode': String (19 chars),
         'title': String,
         'abstract':String
        }
    :return: no return
    """
@app.task(queue="output-results")
def task_output_results(msg):
    """
    This worker will forward results to the outside
    exchange (typically an ADSImportPipeline) to be
    incorporated into the storage

    :param msg: contains the bibcode and the collections:

            {'bibcode': '....',
             'collections': [....]
            }
    :type: adsmsg.OrcidClaims
    :return: no return
    """
    app.forward_message(msg)



if __name__ == "__main__":
    app.start()
