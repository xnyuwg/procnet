from typing import List, Callable
import logging
from procnet.trainer.basic_trainer import BasicTrainer
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from procnet.model.basic_model import BasicModel
from procnet.data_preparer.basic_preparer import BasicPreparer
import torch
from procnet.metric.DocEE_metric import DocEEMetric
from procnet.optimizer.basic_optimizer import BasicOptimizer
from procnet.conf.DocEE_conf import DocEEConfig


class DocEEBasicSeqLabelingTrainer(BasicTrainer):
    def __init__(self,
                 config: DocEEConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 preparer: BasicPreparer,
                 train_loader: DataLoader,
                 dev_loader: DataLoader,
                 test_loader: DataLoader,
                 ):
        super().__init__(config, model, optimizer, preparer, train_loader, dev_loader, test_loader)
        self.result_folder_path = self.result_folder_init(config.model_save_name)
        # uncomment to save model
        # self.model_save_folder_path = self.checkpoint_folder_init(config.model_save_name)
        self.config = config
        self.preparer = preparer

    def train_batch_template(self,
                             model_run_fn: Callable,
                             dataloader: DataLoader,
                             epoch=-1,
                             ):
        self.model.train()
        start_time = time.time()
        batch_step = 0
        epoch_loss = None
        error_num = 0
        with tqdm(dataloader, unit="b", position=0, leave=True) as tqdm_epoch:
            for batch in tqdm(dataloader):
                batch_step += 1
                use_mix_bio = False if epoch <= 2 else True
                loss, res = model_run_fn(self.model, batch, run_eval=False, use_mix_bio=use_mix_bio)
                loss.backward()
                self.optimizer.gradient_update()
                epoch_loss = loss.item() if epoch_loss is None else 0.98 * epoch_loss + 0.02 * loss.item()
                for r in res:
                    if 'error_report' in r and r['error_report'] != '':
                        error_num += 1
        used_time = (time.time() - start_time) / 60
        logging.info('Train Epoch = {}, Time = {:.2f} min, Epoch Mean Loss = {:.4f}, Error Report Num = {}'.format(epoch, used_time, epoch_loss, error_num))

    def eval_batch_template(self,
                            model_run_fn: Callable,
                            score_fn: Callable,
                            dataloader: DataLoader,
                            run_eval=True,
                            epoch=-1,
                            ):
        self.model.eval()
        epoch_loss = 0
        start_time = time.time()
        raw_results: List[dict] = []
        for batch in tqdm(dataloader, unit="b", position=0, leave=True):
            loss, res = model_run_fn(self.model, batch, run_eval=run_eval, use_mix_bio=False)
            epoch_loss += loss.item()
            raw_results += res
        error_reports = set([x['error_report'] for x in raw_results if x['error_report'] != ''])
        if len(error_reports) > 0:
            logging.warning('Eval error: ' + str(error_reports))
        epoch_loss = epoch_loss / len(dataloader)
        score_to_print, score_result = score_fn(raw_results)
        used_time = (time.time() - start_time) / 60
        error_num = sum([1 if r['error_report'] != '' else 0 for r in raw_results])
        logging.info('Eval Epoch = {}, Time = {:.2f} min, Epoch Mean Loss = {:.4f}, Error Report Num = {}, \nScore = {}'.format(epoch, used_time, epoch_loss, error_num, score_to_print))
        return score_result, raw_results

    def train_template(self,
                       model_run_fn: Callable,
                       score_fn: Callable,
                       train_loader: DataLoader = None,
                       dev_loader: DataLoader = None,
                       test_loader: DataLoader = None,
                       ):
        train_loader = self.train_loader if train_loader is None else train_loader
        dev_loader = self.dev_loader if dev_loader is None else dev_loader
        test_loader = self.test_loader if test_loader is None else test_loader
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_formatted = self.epoch_format(epoch, 3)
            self.train_batch_template(model_run_fn, dataloader=train_loader, epoch=epoch)
            # uncomment to save model
            # model_save_path = self.model_save_folder_path / (self.config.model_save_name + '_' + epoch_formatted + '.pth')
            # self.optimizer.save_model(model_save_path)
            logging.info('Eval Epoch = {}, dev:'.format(epoch))
            dev_score_results, dev_raw_results = self.eval_batch_template(model_run_fn, score_fn=score_fn, dataloader=dev_loader, epoch=epoch)
            logging.info('Eval Epoch = {}, test:'.format(epoch))
            test_score_results, test_raw_results = self.eval_batch_template(model_run_fn, score_fn=score_fn, dataloader=test_loader, epoch=epoch)
            final_score_results = {'dev': dev_score_results,
                                   'test': test_score_results,
                                   "epoch": epoch,
                                   }
            score_results_file_name = self.config.model_save_name + '_' + epoch_formatted + '.json'
            self.write_json_file(self.result_folder_path / score_results_file_name, final_score_results)


class DocEETrainer(DocEEBasicSeqLabelingTrainer):
    def __init__(self,
                 config: DocEEConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 preparer: BasicPreparer,
                 metric: DocEEMetric,
                 train_loader: DataLoader,
                 dev_loader: DataLoader,
                 test_loader: DataLoader,
                 ):
        super().__init__(config, model, optimizer, preparer, train_loader, dev_loader, test_loader)
        self.metric = metric
        self.score_fn = metric.the_score_fn

    def model_fn(self, model: BasicModel, batch: list, run_eval: bool, use_mix_bio: bool):
        doc_id, input_ids, input_att_masks, bio_ids, events_labels = (b for b in batch)
        input_ids = input_ids.to(self.device) if isinstance(input_ids, torch.Tensor) else [x.to(self.device) for x in input_ids]
        input_att_masks = input_att_masks.to(self.device) if isinstance(input_att_masks, torch.Tensor) else None
        if run_eval:
            model_res = model(inputs_ids=input_ids,
                              inputs_att_masks=input_att_masks,
                              events_labels=events_labels,
                              )
        else:
            bio_ids_run = bio_ids.to(self.device) if isinstance(bio_ids, torch.Tensor) else [x.to(self.device) for x in bio_ids]
            model_res = model(inputs_ids=input_ids,
                              inputs_att_masks=input_att_masks,
                              events_labels=events_labels,
                              bios_ids=bio_ids_run,
                              use_mix_bio=use_mix_bio,
                              )
        loss, result = model_res
        if isinstance(bio_ids, torch.Tensor):
            BIO_ans = bio_ids.view(-1).detach().cpu().numpy().tolist()
        else:
            BIO_ans = torch.cat(bio_ids, dim=0).view(-1).detach().cpu().numpy().tolist()
        assert len(BIO_ans) == len(result['BIO_pred'])
        events_label = events_labels
        other_record = {
               'doc_id': doc_id,
               'BIO_ans': BIO_ans,
               'event_ans': events_label,
               }
        result.update(other_record)
        return loss, [result]

    def train(self):
        self.train_template(model_run_fn=self.model_fn,
                            score_fn=self.score_fn,
                            )

    def eval(self,
             test_loader: DataLoader = None,
             true_bio: bool = False,
             ):
        test_loader = self.test_loader if test_loader is None else test_loader
        if true_bio:
            score_result, raw_results = self.eval_batch_template(model_run_fn=self.model_fn,
                                                                 score_fn=self.score_fn,
                                                                 dataloader=test_loader,
                                                                 epoch='Test',
                                                                 run_eval=False,
                                                                 )
        else:
            score_result, raw_results = self.eval_batch_template(model_run_fn=self.model_fn,
                                                                 score_fn=self.score_fn,
                                                                 dataloader=test_loader,
                                                                 epoch='Test',
                                                                 )
        return score_result, raw_results
