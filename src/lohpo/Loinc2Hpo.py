from dataclasses import dataclass
from typing import Dict 

from .Annotation import * 
from .Code import *
from .CodeSystemConvertor import * 
from .LohpoError import *


@dataclass
class Loinc2Hpo:
    annotations: Dict[LoincId, Annotation]
    converter: CodeSystemConvertor

    def query(self, loincId: LoincId, testResult: Code) -> HpoTerm4TestOutcome:
        annotation = self.annotations.get(loincId, None)
        if annotation is None:
            raise LohpoError(f"LoincId {loincId} not found")

        result = annotation.candidateHpoTerms.get(testResult)
        if result is None:
            raise LohpoError(f"Code {testResult} at LoincId {loincId} not found")

        return result
