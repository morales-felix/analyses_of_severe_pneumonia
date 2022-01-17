from enum import Enum, auto

from .Code import *
from .LohpoError import *

 
class InternalCode(Enum):
    A = auto()
    L = auto()
    N = auto()
    H = auto()
    NP = auto()
    P = auto()
    U = auto()
    NEG = auto()
    POS = auto()

    @staticmethod
    def from_code(code):
        try:
            return InternalCode.__members__[code]
        except KeyError:
            raise LohpoError(f"Cannot recognize the code: {code}")

    def display(self):
        values = {
            InternalCode.A: "abnormal",
            InternalCode.L: "below normal range",
            InternalCode.N: "within normal range",
            InternalCode.H: "above normal range",
            InternalCode.NP: "not present",
            InternalCode.P: "present",
            InternalCode.U: "unknown code",
            InternalCode.NEG: "not present",
            InternalCode.POS: "present"
        }
        return values.get(self, "?")

    @property
    def lohpo_code(self):
        SYSTEMNAME = "FHIR"
        if self == InternalCode.A: return Code(SYSTEMNAME, "A", "abnormal")
        elif self == InternalCode.L: return Code(SYSTEMNAME, "L", "low")
        elif self == InternalCode.N: return Code(SYSTEMNAME, "N", "normal")
        elif self == InternalCode.H: return Code(SYSTEMNAME, "H", "high")
        elif self == InternalCode.U: return Code(SYSTEMNAME, "U", "unknown")
        elif self == InternalCode.NEG: return Code(SYSTEMNAME, "NEG", "absent")
        elif self == InternalCode.POS: return Code(SYSTEMNAME, "POS", "present")
        else: return None