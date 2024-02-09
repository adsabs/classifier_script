

# from builtins import str
# from .models import ClaimsLog, Records, AuthorInfo, ChangeLog
# from ClassiferPipeline.models import ScoreTable, OverrideTable, FinalCollectionTable
import ClassifierPipeline.models as models
# import ClassifierPipeline.app as app_module
from adsputils import get_date, ADSCelery, u2asc
# from ADSOrcid import names
# from ADSOrcid.exceptions import IgnorableException
# from celery import Celery
# from contextlib import contextmanager
# from dateutil.tz import tzutc
# from sqlalchemy import and_
# from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
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


class SciXClassifierCelery(ADSCelery):


    def __init__(self, *args, **kwargs):
        pass

    def index_record(self, record):
        """
        Sasves a record into a database

        :param: record- dictionar
        :return: boolean - whether record successfuly added
                to the database
        """
        print('Indexing record in index_record')
    # id = Column(Integer, primary_key=True)
    # bibcode = Column(String(19), unique=True)
    # scores = Column(Text)
    # created = Column(UTCDateTime, default=get_date)

        scores = {'scores': {cat:score for cat, score in zip(record['categories'], record['scores'])},
                  'earth_science_adjustment': record['earth_science_adjustment'],
                  'collections': record['collections']}
        
        score_table = models.ScoreTable(bibcode=record['bibcode'], 
                                 scores=scores)

        import pdb; pdb.set_trace()

        # res = []
        with self.session_scope() as session:

            session.add(score_table)
            session.commit()
            
            # for c in claims:
            #     if isinstance(c, ClaimsLog):
            #         claim = c
            #     else:
            #         claim = self.create_claim(**c)
            #     if claim:
            #         session.add(claim)
            #         res.append(claim)
            # session.commit()
            # res = [x.toJSON() for x in res]
        # return res


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



