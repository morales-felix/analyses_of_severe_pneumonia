from dataclasses import dataclass, field
import datetime
from typing import Dict

from .Code import *
from .HpoTerm4TestOutcome import *
from .InternalCode import *
from .LoincId import *
from .LoincScale import *


# This class is responsible for managing the annotation information for a
# LOINC coded lab test. In essence, it uses a map to record mappings from an
# interpretation code to a HPO term (plus a negation boolean value). The key
# of this map, {@link org.monarchinitiative.loinc2hpocore.codesystems.Code},
# must have two parts (code system and
# actual code). We use a set of internal codes for the annotation, which is
# a subset of FHIR codes:
#
# L(ow)                -> Hpo term
# A(bnormal)/N(ormal)  -> Hpo term
# H(igh)               -> Hpo term
# For Ord types with a "Presence" or "Absence" outcome:
# POS(itive)           -> Hpo term
# Neg(ative)           -> Hpo term
#
# It is also okay to annotate with code from any coding system, for example,
# one could use SNOMED concepts.
 
@dataclass
class Annotation:
    loincId: LoincId 
    loincScale: LoincScale 
    candidateHpoTerms: Dict[Code, HpoTerm4TestOutcome] = field(default_factory=dict)
    createdOn: datetime.datetime = None
    createdBy: str = None
    lastEditedOn: datetime.datetime = None 
    lastEditedBy: str = None
    version: float = 0.0
    note: str = None
    flag: bool = False 

    # A convenient method to show hpo term for low
    def whenValueLow(self) -> TermId:
        code = self.candidateHpoTerms.get(InternalCode.L.lohpo_code, None)
        if code is not None:
            return code.id 
        return None
        
    # A convenient method to show hpo term for normal (Qn) or negative (Ord)
    def whenValueNormalOrNegative(self) -> TermId:
        code = self.candidateHpoTerms.get(InternalCode.N.lohpo_code, None)
        if code is not None:
            return code.id 
        code = self.candidateHpoTerms.get(InternalCode.NEG.lohpo_code, None)
        if code is not None:
            return code.id
        return None

    # A convenient method to show hpo term for high (Qn) or positive (Ord)
    def whenValueHighOrPositive(self) -> TermId:
        code = self.candidateHpoTerms.get(InternalCode.H.lohpo_code, None)
        if code is not None:
            return code.id 
        code = self.candidateHpoTerms.get(InternalCode.POS.lohpo_code, None)
        if code is not None:
            return code.id 
        return None

    def __str__(self):
        result = []
        for c, t in self.candidateHpoTerms.items():
            line = f"{self.loincId}\t{self.loincScale.name}"
            line += f"\t{c.system}\t{c.code}"
            line += f"\t{t.id.value}\t{'true' if t.isNegated else 'false'}"
            line += f"\t{self.note or ''}\t{'true' if self.flag else 'false'}"
            line += f"\t{self.version:0.1f}"
            line += f"\t{'NA' if self.createdOn is None else self.createdOn.isoformat()}"
            line += f"\t{self.createdBy or 'NA'}"
            line += f"\t{'NA' if self.lastEditedOn is None else self.lastEditedOn.isoformat()}"
            line += f"\t{self.lastEditedBy or 'NA'}"
            result.append(line)
        return "\n".join(result)