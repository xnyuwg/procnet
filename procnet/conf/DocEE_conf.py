from dataclasses import dataclass
from procnet.conf.basic_conf import BasicConfig


@dataclass
class DocEEConfig(BasicConfig):
    node_size: int = None
    proxy_slot_num: int = None
