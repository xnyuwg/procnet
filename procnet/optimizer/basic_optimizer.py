import logging
from transformers import AdamW, get_scheduler
import torch.nn.utils
from procnet.conf.basic_conf import BasicConfig


class BasicOptimizer:
    def __init__(self,
                 config: BasicConfig,
                 model,
                 ):
        self.model = model
        self.config = config
        self.optimizer = None
        self.scheduler = None
        self.optimizing_no_decay = ['bias,LayerNorm.bias', 'LayerNorm.weight']
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        self.slow_para = None
        self.gradient_accumulation_steps = None
        self.current_step = 0

        self.learning_rate_slow = config.learning_rate_slow
        self.learning_rate_fast = config.learning_rate_fast

        self.create_optimizer(learning_rate_slow=config.learning_rate_slow, learning_rate_fast=config.learning_rate_fast)

    def create_optimizer(self, learning_rate_slow: float, learning_rate_fast: float) -> AdamW:
        no_decay_para = self.optimizing_no_decay
        if self.slow_para is None:
            logging.debug('The slow_para not been assigned')
            slow_para = []
        else:
            slow_para = self.slow_para
        logging.debug('get fast para {} and no decay para {}'.format(slow_para, no_decay_para))
        named_para = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_para if not any(nd in n for nd in no_decay_para) and any(nd in n for nd in slow_para)],
             'weight_decay': self.weight_decay, 'lr': learning_rate_slow},
            {'params': [p for n, p in named_para if any(nd in n for nd in no_decay_para) and any(nd in n for nd in slow_para)],
             'weight_decay': 0.0, 'lr': learning_rate_slow},
            {'params': [p for n, p in named_para if not any(nd in n for nd in no_decay_para) and not any(nd in n for nd in slow_para)],
             'weight_decay': self.weight_decay, 'lr': learning_rate_fast},
            {'params': [p for n, p in named_para if any(nd in n for nd in no_decay_para) and not any(nd in n for nd in slow_para)],
             'weight_decay': 0.0, 'lr': learning_rate_fast}
        ]
        logging.debug('Model Slow learning rate: {}'.format(
            [n for n, p in named_para if any(nd in n for nd in slow_para)]
        ))
        logging.debug('Model Fast learning rate: {}'.format(
            [n for n, p in named_para if not any(nd in n for nd in slow_para)]
        ))
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate_slow)
        self.optimizer = optimizer
        return optimizer

    def create_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            raise Exception("Please init the optimizer first")
        num_warmup_steps = int(num_training_steps * 0.05)
        scheduler = get_scheduler('linear', self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        self.scheduler = scheduler
        return scheduler

    def prepare_for_all(self, learning_rate_slow: float, learning_rate_fast: float, num_training_steps: int,
                             gradient_accumulation_steps: int = None, init_step: int = 0):
        self.create_optimizer(learning_rate_slow=learning_rate_slow, learning_rate_fast=learning_rate_fast)
        self.create_scheduler(num_training_steps=num_training_steps)
        self.gradient_updater_init(gradient_accumulation_steps=gradient_accumulation_steps, init_step=init_step)

    def prepare_for_train(self, num_training_steps: int, gradient_accumulation_steps: int = None, init_step: int = 0):
        self.create_scheduler(num_training_steps=num_training_steps)
        self.gradient_updater_init(gradient_accumulation_steps=gradient_accumulation_steps, init_step=init_step)

    def gradient_updater_init(self, gradient_accumulation_steps: int = None, init_step: int = 0):
        if gradient_accumulation_steps is None:
            logging.warning('The gradient_accumulation_steps not been assigned! please check the code is correct!')
            self.gradient_accumulation_steps = 1
        else:
            self.gradient_accumulation_steps = gradient_accumulation_steps
        self.model.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_step = init_step

    def gradient_update(self, step: int = None):
        """ only step gradient, not gradient backward """
        self.current_step += 1
        if step is None:
            step = self.current_step
        else:
            if step != self.current_step:
                logging.warning('given step not equals to the step stored in the model. Set the step in the model as the given step')
                self.current_step = step
        if step % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def save_model(self, path):
        logging.info('model save to {} ...'.format(path))
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        logging.info('model load from {} ...'.format(path))
        self.model.load_state_dict(torch.load(path))
