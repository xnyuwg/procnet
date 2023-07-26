import logging
import os
from pathlib import Path


class GlobalConfigManager:
    """ init the config and current path """
    current_path = Path(os.path.split(os.path.realpath(__file__))[0] + '/../../')  # the path of the code .py file
    logging.info("Current Path: {}".format(current_path))

    @classmethod
    def if_not_exist_then_creat(cls, path):
        if not os.path.exists(path):
            logging.info("Path not exist: {}, creating...".format(path))
            os.makedirs(path)

    @classmethod
    def get_dataset_path(cls):
        path = cls.current_path / 'Data'
        return path

    @classmethod
    def get_transformers_cache_path(cls):
        path = cls.current_path / 'Cache' / 'Transformers'
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_model_save_path(cls):
        path = cls.current_path / 'Checkpoint'
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_result_save_path(cls):
        path = cls.current_path / 'Result'
        cls.if_not_exist_then_creat(path)
        return path
