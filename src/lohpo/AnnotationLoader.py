import csv
from importlib import resources
from typing import Dict

from .Annotation import *
from .LohpoError import *
from .LoincId import *
from .LoincScale import *


class AnnotationLoader:
    @staticmethod
    def load() -> Dict[LoincId, Annotation]:
        def convert_date(s):
            return None if s == "NA" else datetime.datetime.fromisoformat(s)

        data = resources.read_text("lohpo.resources", "annotations.tsv")
        model_map = {}
        for row in csv.DictReader(data.splitlines(), delimiter="\t"):
            id = LoincId.from_code(row["loincId"])
            code = Code(row['system'], row["code"])
            mappedTo = HpoTerm4TestOutcome(TermId.from_value(row["hpoTermId"]), row["isNegated"] == "true")

            map = model_map.get(id, None)
            if map is None:
                model_map[id] = Annotation(
                    id,
                    LoincScale.from_str(row["loincScale"]),
                    {code: mappedTo},
                    convert_date(row["createdOn"]),
                    row["createdBy"],
                    convert_date(row["lastEditedOn"]),
                    row["lastEditedBy"],
                    float(row["version"]),
                    row["comment"],
                    row["isFinalized"] == "false"
                )
            else:
                map.candidateHpoTerms[code] = mappedTo
        return model_map