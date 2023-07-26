import logging
from typing import List
from procnet.data_preparer.basic_preparer import BasicPreparer
from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_example.DocEEexample import DocEEDocumentExample
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from procnet.conf.DocEE_conf import DocEEConfig
from procnet.utils.util_string import UtilString


class DocEEPreparer(BasicPreparer):
    def __init__(self,
                 config: DocEEConfig,
                 processor: DocEEProcessor,
                 ):
        super().__init__(model_name=config.model_name)
        self.config = config
        self.seq_label_BIO_tag_set = set()
        self.seq_label_category_set = set()
        self.event_type_label_set = set()
        self.event_role_label_set = set()
        self.SCHEMA = processor.SCHEMA
        self.SCHEMA_KEY_CHN_ENG = processor.SCHEMA_KEY_CHN_ENG
        self.SCHEMA_KEY_ENG_CHN = processor.SCHEMA_KEY_ENG_CHN

        self.all_docs: List[List[DocEEDocumentExample]] = [processor.train_docs, processor.dev_docs, processor.test_docs]

        [self.tokenize_sentences(x) for x in self.all_docs]

        [[self.longer_sentence_process_simple_cut(doc, self.config.max_len) for doc in docs] for docs in self.all_docs]

        [[self.seq_label_BIO_tags_generate(doc) for doc in one_docs] for one_docs in self.all_docs]

        self.seq_bio_index_to_cate = ['Null'] + sorted(list(self.seq_label_category_set))
        self.seq_bio_cate_to_index = {self.seq_bio_index_to_cate[i]: i for i in range(len(self.seq_bio_index_to_cate))}

        self.seq_BIO_index_to_tag = ['O'] + [x + '-B' for x in self.seq_bio_index_to_cate[1:]] + [x + '-I' for x in self.seq_bio_index_to_cate[1:]]
        self.seq_BIO_tag_to_index = {self.seq_BIO_index_to_tag[i]: i for i in range(len(self.seq_BIO_index_to_tag))}

        self.seq_bio_tag_index_to_cate_index = {0: 0}
        for cate in self.seq_bio_index_to_cate[1:]:
            cate_index = self.seq_bio_cate_to_index[cate]
            b_index = self.seq_BIO_tag_to_index[cate + '-B']
            i_index = self.seq_BIO_tag_to_index[cate + '-I']
            self.seq_bio_tag_index_to_cate_index[b_index] = cate_index
            self.seq_bio_tag_index_to_cate_index[i_index] = cate_index

        event_type_label_set_from_data = set()
        event_role_label_set_from_data = set()
        for docs in self.all_docs:
            for doc in docs:
                for event in doc.events:
                    for k, v in event.items():
                        if k == 'EventType':
                            event_type_label_set_from_data.add(v)
                        else:
                            event_role_label_set_from_data.add(k)
        for k, v in self.SCHEMA.items():
            self.event_type_label_set.add(k)
            [self.event_role_label_set.add(x) for x in v]
        if self.event_type_label_set != event_type_label_set_from_data:
            logging.warning('event schema type and data not same with schema: {} and: data {}'.format(self.event_type_label_set, event_type_label_set_from_data))
        if self.event_role_label_set != event_role_label_set_from_data:
            logging.warning('event schema role and data not same with schema: {} and: data {}'.format(self.event_role_label_set, event_role_label_set_from_data))
        self.event_type_index_to_type = ['Null'] + sorted(list(self.event_type_label_set))
        self.event_type_type_to_index = {self.event_type_index_to_type[i]: i for i in range(len(self.event_type_index_to_type))}
        logging.debug('num event_type_index_to_type: {}, event_type_index_to_type: {}'.format(len(self.event_type_index_to_type), self.event_type_index_to_type))
        self.event_role_index_to_relation = ['Null'] + sorted(list(self.event_role_label_set))
        self.event_role_relation_to_index = {self.event_role_index_to_relation[i]: i for i in range(len(self.event_role_index_to_relation))}
        logging.debug('num event_role_index_to_relation: {}, event_role_index_to_relation: {}'.format(len(self.event_role_index_to_relation), self.event_role_index_to_relation))

        self.event_schema_index = {}
        for k, v in self.SCHEMA.items():
            new_v = [self.event_role_relation_to_index[x] for x in v]
            new_k = self.event_type_type_to_index[k]
            self.event_schema_index[new_k] = new_v

        self.train_docs = self.all_docs[0]
        self.dev_docs = self.all_docs[1]
        self.test_docs = self.all_docs[2]

        pos_event_num_total = 0
        for doc in self.train_docs:
            pos_event_num_total += len(doc.events)
        all_event_num_total = config.proxy_slot_num * len(self.train_docs)
        neg_event_num_total = all_event_num_total - pos_event_num_total
        self.pos_event_ratio_total = pos_event_num_total / all_event_num_total
        self.neg_event_ratio_total = neg_event_num_total / all_event_num_total

        # bio ratio
        neg_bio_num = 0
        total_bio_num = 0
        for doc in self.train_docs:
            for seq in doc.seq_BIO_tags:
                for x in seq:
                    if x == 'O':
                        neg_bio_num += 1
                    total_bio_num += 1
        pos_bio_num = total_bio_num - neg_bio_num
        self.pos_bio_ratio_total = pos_bio_num / total_bio_num
        self.neg_bio_ratio_total = neg_bio_num / total_bio_num

    def tokenize_sentences(self, docs: List[DocEEDocumentExample]):
        for doc in docs:
            doc.sentences_token = [self.my_tokenize(x) for x in doc.sentences]

    def my_tokenize(self, s: str) -> List[str]:
        r = UtilString.character_tokenize(s)
        return r

    def find_end_pos_for_max_len(self, doc: DocEEDocumentExample, start: int, max_len: int) -> int:
        acc_len = 0
        end = start
        for i in range(start, len(doc.sentences_token)):
            acc_len += len(doc.sentences_token[i])
            if acc_len > max_len:
                break
            else:
                end = i + 1
        if start == end:
            raise Exception('A sentence is more than max_len len! which is {}'.format([len(x) for x in doc.sentences_token]))
        assert sum([len(x) for x in doc.sentences_token[start:end]]) <= max_len
        return end

    def seq_label_BIO_tags_generate(self, doc: DocEEDocumentExample, mode: str = 'BIO'):
        BIO_tags = [['O'] * len(sentence) for sentence in doc.sentences_token]
        for entity in doc.entities:
            cate = entity.field
            self.seq_label_category_set.add(cate)
            B_tag = cate + '-B'
            I_tag = cate + '-I'
            E_tag = cate + '-E'
            for pos in entity.positions:
                if mode == 'BIO':
                    BIO_tags[pos[0]][pos[1]] = B_tag
                    BIO_tags[pos[0]][pos[1] + 1: pos[2]] = [I_tag] * (pos[2] - pos[1] - 1)
                    self.seq_label_BIO_tag_set.add(B_tag)
                    self.seq_label_BIO_tag_set.add(I_tag)
        doc.seq_BIO_tags = BIO_tags

    def longer_sentence_process_simple_cut(self, doc: DocEEDocumentExample, max_len: int):
        all_short = True
        for sentence in doc.sentences_token:
            if len(sentence) > max_len:
                all_short = False
                break
        if all_short:
            return

        cut_record = []
        for i in range(len(doc.sentences_token)):
            if len(doc.sentences_token[i]) > max_len:
                doc.sentences_token[i] = doc.sentences_token[i][:max_len]
                cut_record.append(i)

        for entity in doc.entities:
            all_valid = True
            for pos in entity.positions:
                if pos[0] in cut_record and pos[2] > max_len:
                    all_valid = False
                    break
            if not all_valid:
                new_positions = []
                for pos in entity.positions:
                    if pos[0] in cut_record and pos[2] > max_len:
                        continue
                    else:
                        new_positions.append(pos)
                entity.positions = new_positions

    def get_loader_for_flattened_fragment_before_event(self):
        tokenizer = self.get_auto_tokenizer()

        class MyDataSet(Dataset):
            def padding_to_max_len(self, arrays: List[list], padding_token):
                max_len = max(len(array) for array in arrays)
                new_arrays = []
                for array in arrays:
                    padding_len = max_len - len(array)
                    new_array = array + [padding_token] * padding_len
                    new_arrays.append(new_array)
                for array in new_arrays:
                    assert len(array) == len(new_arrays[0])
                return new_arrays

            def __init__(this,
                         examples: List[DocEEDocumentExample],
                         preparer,
                         tokenizer,
                         ):
                this.examples = examples
                this.tokenizer = tokenizer

            def __len__(this):
                return len(this.examples)

            def __getitem__(this, index):
                example = this.examples[index]

                total_sentence_nums = len(example.sentences_token)

                sub_examples = []
                start = 0
                end = 0
                while end < total_sentence_nums:
                    end = self.find_end_pos_for_max_len(doc=example, start=start, max_len=self.config.max_len)
                    sub_example = example.get_fragment(start_sen=start, end_sen=end)
                    sub_examples.append(sub_example)
                    start = end

                doc_id = example.doc_id

                input_ids = []
                input_att_masks = []
                BIO_ids = []
                for sub_example in sub_examples:
                    input_token = [this.tokenizer.cls_token]
                    for x in sub_example.sentences_token:
                        input_token += x
                    input_id = this.tokenizer.convert_tokens_to_ids(input_token)
                    input_ids.append(input_id)
                    input_att_mask = [1] * len(input_id)
                    input_att_masks.append(input_att_mask)

                    BIO_tags = ['O']
                    for x in sub_example.seq_BIO_tags:
                        BIO_tags += x
                    BIO_id = [self.seq_BIO_tag_to_index[x] for x in BIO_tags]
                    BIO_ids.append(BIO_id)

                events_label = []
                for event in example.events:
                    event_label = {}
                    for k, v in event.items():
                        if k == 'EventType':
                            event_label[k] = self.event_type_type_to_index[v]
                        else:
                            if v is not None:
                                v_id = this.tokenizer.convert_tokens_to_ids(self.my_tokenize(v))
                                event_label[tuple(v_id)] = self.event_role_relation_to_index[k]
                    events_label.append(event_label)

                input_ids = [torch.LongTensor(x) for x in input_ids]
                input_att_masks = [torch.LongTensor(x) for x in input_att_masks]
                BIO_ids = [torch.LongTensor(x) for x in BIO_ids]

                return doc_id, input_ids, input_att_masks, BIO_ids, events_label

        train_dataset = MyDataSet(self.train_docs, self, tokenizer)
        dev_dataset = MyDataSet(self.dev_docs, self, tokenizer)
        test_dataset = MyDataSet(self.test_docs, self, tokenizer)
        logging.info("dataset {} train, {} dev, {} test".format(len(train_dataset), len(dev_dataset), len(test_dataset)))

        def my_collate_fn(batch):
            assert len(batch) == 1
            batch = batch[0]
            doc_ids = batch[0]
            input_ids = batch[1]
            input_att_mask = batch[2]
            BIO_ids = batch[3]
            events_labels = batch[4]
            return doc_ids, input_ids, input_att_mask, BIO_ids, events_labels

        logging.debug("init data loader...")
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=my_collate_fn, shuffle=self.config.data_loader_shuffle)
        dev_loader = DataLoader(dev_dataset, batch_size=1, collate_fn=my_collate_fn, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=my_collate_fn, shuffle=False)

        return train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader
