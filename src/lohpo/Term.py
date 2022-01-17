from dataclasses import dataclass, field
import datetime
from typing import List 

from .TermId import * 

## TODO
@dataclass(frozen=True)
class SimpleXref:
    prefix: str
    value: str

    @staticmethod
    def from_value(value):
        if value is None or len(value) == 0:
            raise LohpoError("simpleXref cannot be null or empty")

        try:
            pos = value.index(":")
            prefix = value[0:pos]
            id = value[pos+1:]
            return SimpleXref(prefix, id)
        except ValueError:
            return SimpleXref("UNKNOWN", value)


class TermSynonym:
    value: str

class Dbxref:
    pass

@dataclass 
class IsA:
    id: TermId 
    description: str 


@dataclass
class Term:
    id: TermId 
    name: str  # The human-readable name of the term.
    altTermIds: List[TermId] = field(default_factory=list)  # Alternative term Ids.
    definition: str = None # The term's definition.
    databaseXrefs: List[SimpleXref] = field(default_factory=list)  # These are the cross-references that go along with the definition. In the case of the HPO, these are often PubMed ids.
    comment: str = None  # The term's comment string.
    subsets: List[str] = field(default_factory=list)  # The names of the subsets that the term is in, empty if none.
    synonyms: List[TermSynonym] = field(default_factory=list)  # The list of term synonyms.
    obsolete: bool = False  # Whether or not the term is obsolete.
    createdBy: str = None  # The term's author name.
    creationDate: datetime.datetime = None  # The term's creation date.
    xrefs: List[Dbxref] = None  # The term's xrefs.

    isA: List[IsA] = field(default_factory=list)  # is_a field
    replacedBy: TermId = None  # replaced_by 
    consider: List[TermId] = field(default_factory=list)  # consider 
    propertyValues: List[str] = field(default_factory=list)  # property_value
