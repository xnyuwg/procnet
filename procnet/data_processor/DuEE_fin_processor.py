import logging
from typing import List, Dict
from procnet.data_processor.basic_processor import BasicProcessor
from procnet.conf.global_config_manager import GlobalConfigManager
from procnet.data_example.DuEEfin_example import DuEEfinDocExample
from procnet.utils.util_data import UtilData


class DuEEfinProcessor(BasicProcessor):
    def __init__(self):
        super().__init__()
        self.data_path = GlobalConfigManager.get_duee_dataset_path()
        logging.info("DuEE_fin Path: {}".format(self.data_path))

        split_path_name = ['train', 'dev']
        all_json_path = [self.data_path / ('duee_fin_' + x + '.json') for x in split_path_name]
        schema_json_path = self.data_path / 'duee_fin_event_schema.json'

        self.all_json = [UtilData.read_raw_jsonl_file(x) for x in all_json_path]
        self.schema_json = UtilData.read_raw_jsonl_file(schema_json_path)

        self.train_num = len(self.all_json[0])
        self.dev_num = len(self.all_json[1])

        self.total_json = self.all_json[0] + self.all_json[1]
        self.all_docs: List[DuEEfinDocExample] = [self.parse_json_one(x) for x in self.total_json]

        self.SCHEMA = self.read_schema(self.schema_json)

    def parse_json_one(self, json) -> DuEEfinDocExample:
        text = json['text']
        if 'event_list' not in json:
            event_list = []
        else:
            event_list = json['event_list']
        doc_id = json['id']
        title = json['title']
        events = []
        for e in event_list:
            event_dict = {}
            event_dict["EventType"] = e['event_type']
            event_dict["trigger"] = e['trigger']
            for arg in e['arguments']:
                event_dict[arg['role']] = arg['argument']
            events.append(event_dict)
        example = DuEEfinDocExample(doc_id=doc_id, title=title, text=text, events=events)
        return example

    def read_schema(self, json):
        schema = {}
        for e in json:
            roles = e['role_list']
            event_role = [r['role'] for r in roles]
            schema[e["event_type"]] = event_role
        return schema

