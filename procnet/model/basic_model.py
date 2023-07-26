import logging
from torch import nn
from transformers import BertModel
from procnet.conf.global_config_manager import GlobalConfigManager
from typing import List
import copy
from scipy.optimize import linear_sum_assignment


class BasicModel(nn.Module):
    o_token = 'O'
    b_suffix = '-B'
    i_suffix = '-I'

    @staticmethod
    def new_bert_model(model_name: str = 'bert-base-uncased') -> BertModel:
        bert = BertModel.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return bert

    @staticmethod
    def find_BIO_spans_positions(spans: list) -> List[list]:
        o_token = BasicModel.o_token
        b_suffix = BasicModel.b_suffix
        if len(spans) == 0:
            return []

        to_read_spans = copy.deepcopy(spans)
        o_split_spans: List[list] = []
        left_index = 0
        while left_index < len(to_read_spans):
            for i in range(left_index, len(to_read_spans)):
                if to_read_spans[i] == o_token:
                    if i == left_index:
                        o_split_spans.append(to_read_spans[i:i + 1])
                        left_index = i + 1
                    else:
                        o_split_spans.append(to_read_spans[left_index:i])
                        left_index = i
                    break
                if i == len(to_read_spans) - 1:
                    o_split_spans.append(to_read_spans[left_index:i + 1])
                    left_index = i + 1
                    break
        assert sum([len(x) for x in o_split_spans]) == len(spans)

        bo_split_spans: List[list] = []
        for oss in o_split_spans:
            if len(oss) == 1:
                bo_split_spans.append(oss)
            else:
                to_read_oss = oss
                while len(to_read_oss) > 0:
                    if len(to_read_oss) == 1:
                        bo_split_spans.append(to_read_oss[:])
                        to_read_oss = []
                    for i in range(len(to_read_oss)):
                        if to_read_oss[i].endswith(b_suffix):
                            if i != 0:
                                bo_split_spans.append(to_read_oss[:i])
                                to_read_oss = to_read_oss[i:]
                                break
                        elif i == len(to_read_oss) - 1:
                            bo_split_spans.append(to_read_oss[:])
                            to_read_oss = []
                            break
        assert sum([len(x) for x in bo_split_spans]) == len(spans)

        boc_split_spans: List[list] = []
        for boss in bo_split_spans:
            if len(boss) == 1:
                boc_split_spans.append(boss)
            else:
                to_read_boss = boss
                while len(to_read_boss) > 0:
                    if len(to_read_boss) == 1:
                        boc_split_spans.append(to_read_boss[:])
                        to_read_boss = []
                    for i in range(1, len(to_read_boss)):
                        if to_read_boss[i-1][:-2] != to_read_boss[i][:-2]:
                            boc_split_spans.append(to_read_boss[:i])
                            to_read_boss = to_read_boss[i:]
                            break
                        elif i == len(to_read_boss) - 1:
                            boc_split_spans.append(to_read_boss[:])
                            to_read_boss = []
                            break
        assert sum([len(x) for x in boc_split_spans]) == len(spans)

        positions = []
        start_len = 0
        for boss in boc_split_spans:
            end_len = start_len + len(boss)
            if len(boss) == 1 and boss[0] == o_token:
                pass
            else:
                positions.append([start_len, end_len])
            start_len = end_len

        return positions

    @staticmethod
    def validify_BIO_span(spans: List[str], positions: List[list], mode='ignore'):
        o_token = BasicModel.o_token
        b_suffix = BasicModel.b_suffix
        spans = copy.deepcopy(spans)
        for pos in positions:
            start, end = pos[0], pos[1]
            if not spans[start].endswith(b_suffix):
                if mode == 'ignore':
                    for i in range(start, end):
                        spans[i] = o_token
                elif mode == 'modify':
                    spans[start] = spans[start][:-2] + b_suffix
        return spans

    @staticmethod
    def event_ordering(matrix):
        cost = matrix
        try:
            row_ind, col_ind = linear_sum_assignment(cost, maximize=False)
        except ValueError:
            logging.error('event_ordering error with matrix: {}'.format(matrix))
            row_ind = [0]
            col_ind = [0]
        assert len(row_ind) == len(col_ind)
        res = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
        min_cost = cost[row_ind, col_ind].sum()
        return res, min_cost

    def __init__(self,
                 gradient_accumulation_steps: int = None):
        super().__init__()
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = None
        self.scheduler = None
        self.optimizing_no_decay = ['bias,LayerNorm.bias', 'LayerNorm.weight']
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        self.slow_para = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
