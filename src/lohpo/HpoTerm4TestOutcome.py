from dataclasses import dataclass

from .TermId import *


@dataclass
class HpoTerm4TestOutcome:
    id: TermId
    isNegated: bool = False
