from typing import List
from procnet.data_preparer.basic_preparer import BasicPreparer
from procnet.model.basic_model import BasicModel
from sklearn.metrics import precision_recall_fscore_support
import time
from procnet.utils.util_structure import UtilStructure
from procnet.utils.util_math import UtilMath


class BasicMetric:
    @staticmethod
    def span_overlap_f1(pred: list, ans: list):
        assert len(ans) == len(pred)
        true_positive = 0
        ans_positive = 0
        pred_positive = 0
        for i in range(len(ans)):
            if ans[i] == pred[i] and ans[i] != BasicModel.o_token:
                true_positive += 1
            if ans[i] != BasicModel.o_token:
                ans_positive += 1
            if pred[i] != BasicModel.o_token:
                pred_positive += 1
        precision = true_positive / pred_positive if pred_positive != 0 else 0
        recall = true_positive / ans_positive if ans_positive != 0 else 0
        f1 = UtilMath.calculate_f1(precision, recall)
        return precision, recall, f1

    @staticmethod
    def span_exact_match_f1(pred: list, ans: list, pred_positions: List[list], ans_positions: List[list]):
        assert len(ans) == len(pred)
        ans_positive = len(ans_positions)
        pred_positive = len(pred_positions)
        true_positive = 0

        pred_positions_hash = set()
        for pos in pred_positions:
            pred_positions_hash.add(str(pos[0]) + '-' + str(pos[1]))
        ans_positions_hash = set()
        for pos in ans_positions:
            ans_positions_hash.add(str(pos[0]) + '-' + str(pos[1]))

        for pos in pred_positions_hash:
            if pos in ans_positions_hash:
                items = pos.split('-')
                start = int(items[0])
                end = int(items[1])
                equal = True
                for i in range(start, end):
                    if pred[i] != ans[i]:
                        equal = False
                        break
                if equal:
                    true_positive += 1
        precision = true_positive / pred_positive if pred_positive != 0 else 0
        recall = true_positive / ans_positive if ans_positive != 0 else 0
        f1 = UtilMath.calculate_f1(precision, recall)
        return precision, recall, f1

    def __init__(self,
                 preparer: BasicPreparer,):
        self.seq_BIO_index_to_tag = preparer.seq_BIO_index_to_tag
        self.event_role_relation_to_index = preparer.event_role_relation_to_index
        self.event_role_index_to_relation = preparer.event_role_index_to_relation

    def bio_score_fn(self, bio_ans: List[list], bio_pred: List[list]):
        start_time = time.time()

        all_ans = []
        for x in bio_ans:
            all_ans += x

        all_pred = []
        for x in bio_pred:
            prob = x
            hard_select = []
            for p in prob:
                index = UtilStructure.find_max_number_index(p)
                hard_select.append(index)
            all_pred += hard_select

        assert len(all_ans) == len(all_pred)
        final_ans = []
        final_pred = []
        for i in range(len(all_ans)):
            if all_ans[i] != -100:
                final_ans.append(all_ans[i])
                final_pred.append(all_pred[i])

        BIO_ans = [self.seq_BIO_index_to_tag[x] for x in final_ans]
        BIO_pred = [self.seq_BIO_index_to_tag[x] for x in final_pred]

        BIO_pred_positions = BasicModel.find_BIO_spans_positions(BIO_pred)

        BIO_pred_valid_ignore = BasicModel.validify_BIO_span(BIO_pred, BIO_pred_positions, 'ignore')

        BIO_micro_precision, BIO_micro_recall, BIO_micro_f1, _ = precision_recall_fscore_support(final_pred, final_ans, average='micro')

        raw_span_precision, raw_span_recall, raw_span_f1 = self.span_overlap_f1(BIO_pred, BIO_ans)

        valid_ignore_span_precision, valid_ignore_span_recall, valid_ignore_span_f1 = self.span_overlap_f1(BIO_pred_valid_ignore, BIO_ans)

        BIO_pred_valid_ignore_positions = BasicModel.find_BIO_spans_positions(BIO_pred_valid_ignore)
        BIO_ans_positions = BasicModel.find_BIO_spans_positions(BIO_ans)
        em_ignore_precision, em_ignore_recall, em_ignore_f1 = self.span_exact_match_f1(BIO_pred_valid_ignore, BIO_ans, BIO_pred_valid_ignore_positions, BIO_ans_positions)

        used_time = (time.time() - start_time) / 60

        score_result = {'raw_span': {'precision': raw_span_precision,
                                     'recall': raw_span_recall,
                                     'f1': raw_span_f1},
                        'valid_span': {'precision': valid_ignore_span_precision,
                                       'recall': valid_ignore_span_recall,
                                       'f1': valid_ignore_span_f1},
                        'valid_exact_match': {'precision': em_ignore_precision,
                                              'recall': em_ignore_recall,
                                              'f1': em_ignore_f1}
                        }

        to_print = "BIO EVALUATION: Used Time = {:.2f} min, " \
                   "\nBIO Micro Accuracy = {:.4f}, " \
                   "\nRaw-Span Precision = {:.4f}, Raw-Span Recall = {:.4f}, Raw-Span F1 = {:.4f}, " \
                   "\nValid-Span Precision = {:.4f}, Valid-Span Recall = {:.4f}, Valid-Span F1 = {:.4f}, " \
                   "\nSpan-Exact-Match Precision = {:.4f}, Span-Exact-Match Recall = {:.4f}, Span-Exact-Match F1 = {:.4f}, " \
                   "".format(
                    used_time,
                    BIO_micro_f1,
                    raw_span_precision, raw_span_recall, raw_span_f1,
                    valid_ignore_span_precision, valid_ignore_span_recall, valid_ignore_span_f1,
                    em_ignore_precision, em_ignore_recall, em_ignore_f1,
        )
        return to_print, score_result
