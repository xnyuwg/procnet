import torch
import torch.utils.data
from procnet.model.basic_model import BasicModel
from procnet.data_preparer.basic_preparer import BasicPreparer
from procnet.conf.global_config_manager import GlobalConfigManager
from procnet.optimizer.basic_optimizer import BasicOptimizer
import json
import os


class BasicTrainer:
    @staticmethod
    def write_json_file(file_name, data):
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def result_folder_init(folder_name):
        path = GlobalConfigManager.get_result_save_path() / folder_name
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def checkpoint_folder_init(folder_name):
        path = GlobalConfigManager.get_model_save_path() / folder_name
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def epoch_format(epoch, length):
        epoch_formatted = str(epoch)
        epoch_formatted = '0' * (length - len(epoch_formatted)) + epoch_formatted
        return epoch_formatted

    def __init__(self,
                 config,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 preparer: BasicPreparer,
                 train_loader: torch.utils.data.DataLoader,
                 dev_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 ):
        self.device = config.device
        self.model: BasicModel = model
        self.optimizer: BasicOptimizer = optimizer
        self.preparer: BasicPreparer = preparer
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.dev_loader: torch.utils.data.DataLoader = dev_loader
        self.test_loader: torch.utils.data.DataLoader = test_loader
        num_training_steps = (len(train_loader) * config.max_epochs) // config.gradient_accumulation_steps + 1
        self.optimizer.prepare_for_train(num_training_steps=num_training_steps,
                                         gradient_accumulation_steps=config.gradient_accumulation_steps)
        if config.model_load_name is not None:
            self.optimizer.load_model(GlobalConfigManager.get_model_save_path() / config.model_load_name)
