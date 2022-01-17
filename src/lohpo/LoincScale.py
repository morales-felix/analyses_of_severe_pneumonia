from enum import Enum, auto


class LoincScale(Enum):
    Qn = auto()
    Ord = auto()
    OrdQn = auto()
    Nom = auto()
    Nar = auto()
    Multi = auto()
    Doc = auto()
    Set = auto()
    Unknown = auto()

    @staticmethod
    def from_str(s):
        s = s.lower()
        values = {
            "qn": LoincScale.Qn,
            "ord": LoincScale.Ord,
            "ordqn": LoincScale.OrdQn,
            "nom": LoincScale.Nom,
            "nar": LoincScale.Nar,
            "multi": LoincScale.Multi,
            "doc": LoincScale.Doc,
            "set": LoincScale.Set
        }
        return values.get(s, LoincScale.Unknown)