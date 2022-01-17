from dataclasses import dataclass


@dataclass(frozen=True)
class Code:
    system: str = ""
    code: str = ""
    display: str = ""

    def __str__(self):
        return f"System: {self.system}, Code: {self.code}, Display: {self.display}"
        