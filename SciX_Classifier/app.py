

# from builtins import str
# from .models import ClaimsLog, Records, AuthorInfo, ChangeLog
# from adsputils import get_date, ADSCelery, u2asc
# from ADSOrcid import names
# from ADSOrcid.exceptions import IgnorableException
# from celery import Celery
# from contextlib import contextmanager
# from dateutil.tz import tzutc
# from sqlalchemy import and_
# from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session
# from sqlalchemy.orm import sessionmaker
# import cachetools
# import datetime
# import json
# import os
# import random
# import time
# import traceback

# global objects; we could make them belong to the app object but it doesn't seem necessary
# unless two apps with a different endpint/config live along; TODO: move if necessary
# cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# orcid_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# ads_cache = cachetools.TTLCache(maxsize=1024, ttl=3600, timer=time.time, missing=None, getsizeof=None)
# bibcode_cache = cachetools.TTLCache(maxsize=2048, ttl=3600, timer=time.time, missing=None, getsizeof=None)

# ALLOWED_STATUS = set(['claimed', 'updated', 'removed', 'unchanged', 'forced', '#full-import'])



def clear_caches():
    """Clears all the module caches."""
    cache.clear()
    classifier_cache.clear()
    ads_cache.clear()
    bibcode_cache.clear()


class ADSClassifierCelery(ADSCelery):


    def __init__(self, *args, **kwargs):
        pass

    def score_record_collections(self, record, classifier):
        """
        Given a record and a classifier, score the record
        and return a list of scores

        :param: record - Records object
        :param: classifier - Classifier object

        :return: list of scores
        """
        pass


    def postprocess_classifier_scores(self, record, scores):
        """
        Given a record and a list of scores, postprocess
        the scores and return a list of collections

        :param: record - Records object
        :param: scores - list of scores

        :return: list of scores
        """
        pass



