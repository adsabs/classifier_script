# -*- coding: utf-8 -*-

from builtins import str
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.types import Enum
import json
import sys
from adsputils import get_date, UTCDateTime

Base = declarative_base()


class ScoreInfo(Base):
    __tablename__ = 'scores'
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19), unique=True)
    scores = Column(Text)
    created = Column(UTCDateTime, default=get_date)

class OverrideInfo(Base):
    __tablename__ = 'overrides'
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19), unique=True)
    override = Column(Text)
    created = Column(UTCDateTime, default=get_date)

class FinalCollection(Base):
    __tablename__ = 'final_collection'
    id = Column(Integer, primary_key=True)
    bibcode = Column(String(19), unique=True)
    collection = Column(Text)
    created = Column(UTCDateTime, default=get_date)

