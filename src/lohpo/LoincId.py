from dataclasses import dataclass

from .LohpoError import *


@dataclass(frozen=True)
class LoincId:
    num: int 
    suffix: int

    @staticmethod
    def from_code(loinc_code, has_prefix=False):
        if has_prefix:
            chunks = loinc_code.split(":")
            if len(chunks) < 2:
                raise LohpoError(f"Prefix not found in {loinc_code}")
            loinc_code = chunks[1]

        try:    
            dash_pos = loinc_code.index("-")
        except ValueError:
            raise LohpoError(f"No dash found in {loinc_code}")

        if dash_pos == 0:
            raise LohpoError(f"No numerical part found")
        
        try:
            num = int(loinc_code[:dash_pos])
        except ValueError:
            raise LohpoError(f"Unable to parse numerical part of {loinc_code}")
        try:
            suffix = int(loinc_code[dash_pos+1:])
        except ValueError:
            raise LohpoError(f"Unable to parse suffix of {loinc_code}")

        return LoincId(num, suffix)

    def __str__(self):
        return f"{self.num}-{self.suffix}"