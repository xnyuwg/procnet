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

    @staticmethod
    def read_raw_jsonl_file(file_name, verbose=True) -> list:
        if verbose:
            logging.info("reading jsonl from: {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as file:
            file_content = [json.loads(line) for line in file]
        return file_content

    @staticmethod
    def write_json_file(file_name, data, verbose=True):
        if verbose:
            logging.info("writing json to: {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)



