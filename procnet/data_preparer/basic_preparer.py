import logging
from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer
from procnet.conf.global_config_manager import GlobalConfigManager


class BasicPreparer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.__bert_tokenizer = None
        self.__auto_tokenizer = {}

        self.seq_BIO_index_to_tag = None
        self.seq_BIO_tag_to_index = None
        self.event_type_index_to_type = None
        self.event_type_type_to_index = None
        self.event_role_index_to_relation = None
        self.event_role_relation_to_index = None
        self.SCHEMA = {}
        self.event_schema_index = {}

    def get_auto_tokenizer(self,  model_name: str = None) -> PreTrainedTokenizer:
        if model_name is None:
            model_name = self.model_name
        if model_name not in self.__auto_tokenizer:
            logging.debug("init {} auto tokenizer".format(model_name))
            if model_name == "fnlp/bart-base-chinese":
                tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            elif model_name == "hfl/chinese-roberta-wwm-ext":
                tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            self.__auto_tokenizer[model_name] = tokenizer
        return self.__auto_tokenizer[model_name]



