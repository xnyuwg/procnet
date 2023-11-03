import importlib
import torch
import logging
import argparse
from procnet.utils.util_string import UtilString
from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_preparer.DocEE_preparer import DocEEPreparer
from procnet.model.DocEE_proxy_node_model import DocEEProxyNodeModel
from procnet.optimizer.basic_optimizer import BasicOptimizer
from procnet.trainer.DocEE_proxy_node_trainer import DocEETrainer
from procnet.metric.DocEE_metric import DocEEMetric
from procnet.conf.DocEE_conf import DocEEConfig
importlib.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--run_save_name", type=str, required=True, help="The save name of this run")
    arg_parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    arg_parser.add_argument("--epoch", type=int, default=50, help="Training epochs")
    arg_parser.add_argument("--read_pseudo", type=str, default=False, required=False, help="If read pseudo data")
    args = arg_parser.parse_args(args=in_args)
    args.read_pseudo = UtilString.str_to_bool(args.read_pseudo)
    return args


def get_config(args) -> DocEEConfig:
    config = DocEEConfig()
    config.model_save_name = args.run_save_name
    config.node_size = 512
    config.proxy_slot_num = 16
    config.gradient_accumulation_steps = args.batch_size
    config.max_epochs = args.epoch
    config.data_loader_shuffle = True
    config.model_name = "hfl/chinese-roberta-wwm-ext"
    config.device = torch.device('cuda')
    return config


def run(args):
    config = get_config(args)
    logging.info('save_name = {}'.format(config.model_save_name))
    dee_pro = DocEEProcessor(args.read_pseudo)
    dee_pre = DocEEPreparer(config=config, processor=dee_pro)
    pre_data = dee_pre.get_loader_for_flattened_fragment_before_event()
    train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader = pre_data
    metric = DocEEMetric(preparer=dee_pre)
    model = DocEEProxyNodeModel(config=config, preparer=dee_pre)
    model.to(config.device)
    optimizer = BasicOptimizer(config=config, model=model)
    trainer = DocEETrainer(config=config,
                           model=model,
                           optimizer=optimizer,
                           preparer=dee_pre,
                           metric=metric,
                           train_loader=train_loader,
                           dev_loader=dev_loader,
                           test_loader=test_loader,
                           )
    trainer.train()


if __name__ == '__main__':
    arg = parse_args()
    run(arg)
