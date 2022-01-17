from importlib import resources

from .Term import *


class HpoTermListLoader:
    @staticmethod
    def load() -> List[Term]:
        data = resources.read_text("lohpo.resources", "hp.obo.txt")

        header = []
        terms = []
        term_properties = set()
        current_term = None
        reading_header = True
        for line in data.splitlines():
            line = line.strip()
            if len(line) == 0:
                if reading_header:
                    reading_header = False
                continue

            if line == "[Term]":
                if current_term is not None:
                    terms.append(current_term)
                current_term = []
                continue

            pos = line.index(":")
            name = line[0:pos].strip()
            value = line[pos+1:].strip()

            if reading_header:
                header.append((name, value))
            else:
                current_term.append((name, value))
                term_properties.add(name)

        if len(current_term) > 0:
            terms.append(current_term)

        result = []
        for term in terms:
            id = None
            for key, value in term:
                if key == "id":
                    id = TermId.from_value(value)
                    break
            if id is None:
                continue

            new_term = None
            for key, value in term:
                if key == "name":
                    if new_term is not None:
                        raise LohpoError(f"Structure not valid at hp.obo.txt")
                    new_term = Term(id, value)
                elif key == "alt_id":
                    new_id = TermId.from_value(value)
                    new_term.altTermIds.append(new_id)
                elif key == "def":
                    new_term.definition = value 
                elif key == "xref":
                    simpleXref = SimpleXref.from_value(value)
                    new_term.databaseXrefs.append(simpleXref)
                elif key == "comment":
                    new_term.comment = value 
                elif key == "subset":
                    new_term.subsets.append(value)
                elif key == "synonym":
                    new_term.synonyms.append(value)
                elif key == "is_obsolete":
                    new_term.obsolete = value == "true"
                elif key == "created_by":
                    new_term.createdBy = value 
                elif key == "creation_date":
                    new_term.createdDate = datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
                elif key == "is_a":
                    params = value.split(" ! ")
                    if len(params) != 2:
                        raise LohpoError(f"Cannot find the is_a format for id {id}")
                    isA_id = TermId.from_value(params[0])
                    new_term.isA.append(IsA(isA_id, params[1]))
                elif key == "replaced_by":
                    replacedBy = TermId.from_value(value)
                    if new_term.replacedBy is not None:
                        raise LohpoError(f"Replaced by contains multiple values {id}")
                    new_term.replacedBy = replacedBy
                elif key == "consider":
                    consider = TermId.from_value(value)
                    new_term.consider.append(consider)
                elif key == "property_value":
                    new_term.propertyValues.append(value)
                elif key == "id":
                    pass
                else:
                    raise LohpoError(f"Key {key} not found")

            result.append(new_term)
        d = {t.id: t for t in result}
        if len(d) != len(result):
            raise LohpoError("Duplicated TermId!")
        return header, d