from typing import List
import copy
from procnet.metric.basic_metric import BasicMetric
from procnet.data_preparer.basic_preparer import BasicPreparer
import time
from procnet.dee import dee_metric
from procnet.utils.util_structure import UtilStructure


class DocEEMetric(BasicMetric):
    def __init__(self,
                 preparer: BasicPreparer,):
        super(DocEEMetric, self).__init__(preparer=preparer)
        self.event_schema_index = preparer.event_schema_index
        self.event_type_type_to_index = preparer.event_type_type_to_index
        self.event_type_index_to_type = preparer.event_type_index_to_type
        self.event_role_relation_to_index = preparer.event_role_relation_to_index
        self.event_role_index_to_relation = preparer.event_role_index_to_relation
        self.seq_BIO_index_to_tag = preparer.seq_BIO_index_to_tag
        self.event_schema = preparer.SCHEMA
        self.event_null_type_index = preparer.event_type_type_to_index['Null']
        self.event_null_relation_index = preparer.event_role_relation_to_index['Null']

    def the_score_fn(self, results: List[dict]):
        # loss
        total_num = len(results)
        mean_loss = sum([x['loss'] for x in results]) / total_num
        loss_to_print = "Loss = {:.4f}, ".format(mean_loss)
        # bio
        bio_ans = [x['BIO_ans'] for x in results]
        bio_pred = [x['BIO_pred'] for x in results]
        bio_to_print, bio_score_results = self.bio_score_fn(bio_ans=bio_ans, bio_pred=bio_pred)
        # event
        if 'event_ans' in results[0] and 'event_pred' in results[0]:
            # events_ans: [[{'EventType': 3, (104, 105, 106): 4, 'entity': 7}, {}], [{}, {}]]
            # events_pred: [[{'EventType': [0.1, 0.3, 0.5], '(3, 4, 103)': [0.2, 0.4 0.2]}, {}, {}], [], []]
            event_ans_all = [x['event_ans'] for x in results]
            event_pred_all = [x['event_pred'] for x in results]
            event_ans_single = []
            event_pred_single = []
            event_ans_multi = []
            event_pred_multi = []
            assert len(event_ans_all) == len(event_pred_all)
            for i in range(len(event_ans_all)):
                ea = event_ans_all[i]
                ep = event_pred_all[i]
                if len(ea) <= 1:
                    event_ans_single.append(ea)
                    event_pred_single.append(ep)
                else:
                    event_ans_multi.append(ea)
                    event_pred_multi.append(ep)

            all_dee_to_print = []
            all_dee_score_results = []
            for event_ans, event_pred in zip([event_ans_all, event_ans_single, event_ans_multi], [event_pred_all, event_pred_single, event_pred_multi]):
                dee_to_print, dee_score_results = self.dee_score_fn(event_ans, event_pred)
                all_dee_to_print.append(dee_to_print)
                all_dee_score_results.append(dee_score_results)
            # dee_to_print = "All event:" + all_dee_to_print[0] + "\nSingle Event:" + all_dee_to_print[1] + "\nMulti Event:" + all_dee_to_print[2]
            dee_to_print = all_dee_to_print[0]
            dee_score_results = {'all_event': all_dee_score_results[0],
                                 'single_event': all_dee_score_results[1],
                                 'multi_event': all_dee_score_results[2],
                                 }
        else:
            dee_to_print, dee_score_results = "", {}

        to_print = loss_to_print + '\n' + dee_to_print
        final_score_results = {
                               'loss': mean_loss,
                               'bio': bio_score_results,
                               'event': dee_score_results,
                               }
        return to_print, final_score_results

    def dee_score_fn(self, events_ans: List[List[dict]], events_pred: List[List[dict]]):
        start_time = time.time()
        type_to_index: dict = copy.deepcopy(self.event_type_type_to_index)
        type_to_index.pop('Null')
        type_to_index = {k: v - 1 for k, v in type_to_index.items()}
        index_to_type = copy.deepcopy(self.event_type_index_to_type)
        index_to_type = index_to_type[1:]
        index_to_type = {i: index_to_type[i] for i in range(len(index_to_type))}
        role_to_index: dict = self.event_role_relation_to_index

        event_num = len(type_to_index)

        event_schema = self.event_schema
        event_type_roles_list = []
        event_type_list = []
        for i in range(event_num):
            event_type = index_to_type[i]
            event_type_list.append(event_type)
            roles = event_schema[event_type]
            event_type_roles_list.append((event_type, roles))

        gold_record_mat_list = []
        for event_ans in events_ans:
            gold_record_mat = [[] for _ in range(event_num)]
            for e_ans in event_ans:
                # {4: (104, 105, 106), 7: 'entity'}
                roles_dict = {v: k for k, v in e_ans.items() if k != 'EventType'}
                event_type = self.event_type_index_to_type[e_ans['EventType']]
                event_type_id = type_to_index[event_type]
                roles_tuple = []
                for i in range(len(event_schema[event_type])):
                    role_name = event_schema[event_type][i]
                    role_index = role_to_index[role_name]
                    if role_index in roles_dict:
                        roles_tuple.append(roles_dict[role_index])
                    else:
                        roles_tuple.append(None)
                roles_tuple = tuple(roles_tuple)
                gold_record_mat[event_type_id].append(roles_tuple)
            gold_record_mat_list.append(gold_record_mat)

        pred_record_mat_list = []
        for event_pred in events_pred:
            pred_record_mat = [[] for _ in range(event_num)]
            for e_pred in event_pred:
                event_type = UtilStructure.find_max_number_index(e_pred['EventType'])
                event_type = self.event_type_index_to_type[event_type]
                if event_type == 'Null':
                    continue
                event_type_id = type_to_index[event_type]

                roles_dict = {}
                for k, v in e_pred.items():
                    if k == 'EventType':
                        continue
                    max_p, index = UtilStructure.find_max_and_number_index(v)
                    if index not in roles_dict:
                        roles_dict[index] = [[k, max_p]]
                    else:
                        roles_dict[index].append([k, max_p])
                for k in roles_dict:
                    best_k = None
                    best_p = -1
                    for x in roles_dict[k]:
                        if x[1] > best_p:
                            best_k, best_p = x[0], x[1]
                    roles_dict[k] = [best_k, best_p]
                for k in roles_dict:
                    roles_dict[k] = roles_dict[k][0]

                roles_tuple = []
                for i in range(len(event_schema[event_type])):
                    role_name = event_schema[event_type][i]
                    role_index = role_to_index[role_name]
                    if role_index in roles_dict:
                        roles_tuple.append(roles_dict[role_index])
                    else:
                        roles_tuple.append(None)
                roles_tuple = tuple(roles_tuple)
                pred_record_mat[event_type_id].append(roles_tuple)
            pred_record_mat_list.append(pred_record_mat)

        score_results = dee_metric.measure_event_table_filling(pred_record_mat_list, gold_record_mat_list, event_type_roles_list, event_type_list)

        used_time = (time.time() - start_time) / 60
        score_results['used_time'] = used_time

        to_print = "dee_metric: Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f}, ".format(
            score_results['micro_precision'], score_results['micro_recall'], score_results['micro_f1'],
        )
        return to_print, score_results
