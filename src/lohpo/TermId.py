from dataclasses import dataclass

from .LohpoError import *


@dataclass(frozen=True)
class TermId:
    prefix: str
    id: str

    @staticmethod 
    def from_value(value):
        if value is None or len(value) == 0:
            raise LohpoError("termId cannot be null or empty")
        try:
            pos = value.index(":")
        except ValueError:
            raise LohpoError(f"TermId construction error: '{value}' does not have a prefix!")

        prefix = value[0:pos]
        id = value[pos+1:]
        return TermId(prefix, id)

    @staticmethod
    def from_prefix_id(prefix, id):
        if prefix is None or len(prefix) == 0:
            raise LohpoError("termPrefix cannot be null or empty")
        if id is None or len(id) == 0:
            raise LohpoError("term id cannot be null or empty")
        return TermId(prefix, id)

    @property
    def value(self):
        return f"{self.prefix}:{self.id}"

    def __str__(self):
        return self.value
