import csv
from importlib import resources

from .Code import *
from .InternalCode import *
from .LohpoError import *

class CodeSystemConvertor:
    def __init__(self):
        v2System = "http://hl7.org/fhir/v2/0078"

        map = {}
        data = resources.read_text("lohpo.resources", "HL7_v2_table0078_to_internal.tsv")
        for row in csv.reader(data.splitlines()[1:], delimiter="\t"):
            if len(row) != 3:
                raise LohpoError(f"The line does not have 3 tab-separated elements: {row}")
            v2Code = Code(v2System, row[0], None)
            internalCode = InternalCode.from_code(row[2]).lohpo_code
            map[v2Code] = internalCode
        self.map = map