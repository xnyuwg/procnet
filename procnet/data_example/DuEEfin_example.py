from dataclasses import dataclass
from typing import List, Dict
import copy


@dataclass
class DuEEfinEntity:
    span: str
    positions: List[list]  # [[sentence_index, start, end]]
    field: str

    def print_all(self) -> str:
        """ print all attributes """
        res = ""
        res += ("span:{}\n".format(self.span))
        res += ("positions:{}\n".format(self.positions))
        res += ("field:{}\n".format(self.field))
        return res


@dataclass
class DuEEfinDocExample:
    doc_id: str
    title: str
    text: str
    events: List[Dict[str, str]]
    sentences: List[str] = None
    entities: List[DuEEfinEntity] = None

    def copy(self):
        return copy.deepcopy(self)
