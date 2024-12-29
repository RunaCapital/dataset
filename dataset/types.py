from datetime import datetime, date
from typing import Any, Iterable, Type

from sqlalchemy import Integer, UnicodeText, Float, BigInteger
from sqlalchemy import String, Boolean, Date, DateTime, Unicode, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeEngine, _Binary

MYSQL_LENGTH_TYPES = (String, _Binary)


class Types(object):
    """A holder class for easy access to SQLAlchemy type names."""

    integer = Integer
    string = Unicode
    text = UnicodeText
    float = Float
    bigint = BigInteger
    boolean = Boolean
    date = Date
    datetime = DateTime

    def __init__(self, is_postgres=None):
        self.json = JSONB if is_postgres else JSON

    def guess(self, samples: Iterable[Any]) -> TypeEngine | Type:
        """Given a list of samples, guess the column type for the field.

        If the first non-null sample is an instance of an SQLAlchemy type,
        the type will be used instead.

        Defaults to 'text' if all values are null.
        Chooses 'text' if there are mixed types in the samples.
        """
        detected_types = set()
        for sample in samples:
            if sample is None:
                continue
            if isinstance(sample, TypeEngine):
                return sample
            if isinstance(sample, bool):
                detected_types.add(self.boolean)
            elif isinstance(sample, int):
                detected_types.add(self.bigint)
            elif isinstance(sample, float):
                detected_types.add(self.float)
            elif isinstance(sample, datetime):
                detected_types.add(self.datetime)
            elif isinstance(sample, date):
                detected_types.add(self.date)
            elif isinstance(sample, dict):
                detected_types.add(self.json)
            else:
                detected_types.add(self.text)

        if len(detected_types) == 0:
            return self.text
        elif len(detected_types) == 1:
            return detected_types.pop()
        elif self.text in detected_types:
            return self.text
        elif {self.float, self.bigint} == detected_types:
            return self.float
        elif {self.date, self.datetime} == detected_types:
            return self.datetime
        else:
            return self.text
