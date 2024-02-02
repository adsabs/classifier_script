# -*- coding: utf-8 -*-

from builtins import str
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.types import Enum
import json
import sys
from adsputils import get_date, UTCDateTime

Base = declarative_base()


class ScoreTable(Base):
    __tablename__ = 'scores'
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19), unique=True)
    scores = Column(Text)
    created = Column(UTCDateTime, default=get_date)

class OverrideTable(Base):
    __tablename__ = 'overrides'
    id = Column(Integer, primary_key=True)
    score_id = Column(Integer, foreign_key='scores.id')
    # score_id = Column(Integer, foreign_key='ScoreTable.id')
    override = Column(ARRAY(String))
    created = Column(UTCDateTime, default=get_date)

class FinalCollectionTable(Base):
    __tablename__ = 'final_collection'
    id = Column(Integer, primary_key=True)
    score_id = Column(Integer, foreign_key='scores.id')
    # score_id = Column(Integer, foreign_key='ScoreTable.id')
    collection = Column(ARRAY(String))
    created = Column(UTCDateTime, default=get_date)

