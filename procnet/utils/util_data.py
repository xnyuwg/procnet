import logging
import json


class UtilData:
    def __init__(self):
        pass

    @staticmethod
    def read_raw_json_file(file_name) -> dict:
        logging.debug("reading json from: {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data



