from procnet.conf.global_config_manager import GlobalConfigManager
import logging
from procnet.data_preparer.basic_preparer import BasicPreparer
from procnet.data_processor.DuEE_fin_processor import DuEEfinProcessor
from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_example.DuEEfin_example import DuEEfinDocExample, DuEEfinEntity
from procnet.data_example.DocEEexample import DocEEDocumentExample, DocEEEntity
import random
from procnet.utils.util_data import UtilData


class DuEEfinPreparer(BasicPreparer):
    def __init__(self,
                 processor: DuEEfinProcessor,
                 ):
        super().__init__(model_name="hfl/chinese-roberta-wwm-ext")
        self.processor = processor
        self.SCHEMA = processor.SCHEMA
        self.all_docs = processor.all_docs
        self.train_num = processor.train_num
        self.dev_num = processor.dev_num

        [self.text_correct(doc) for doc in self.all_docs]
        self.sentence_sep()
        self.bio_tag_generate_all_by_event()

    def text_correct(self, doc: DuEEfinDocExample):
        doc.text = '原标题：' + doc.title + '。原文：' + doc.text

    def sentence_sep(self):
        for doc in self.all_docs:
            text = doc.text
            is_title = True
            start = 0
            sentences = []
            for i in range(len(text)):
                if is_title and text[i:i + 3] == '来源：':
                    sentences.append(text[start:i])
                    start = i
                    is_title = False
                elif text[i] == '。':
                    sentences.append(text[start:i+1])
                    start = i+1
            if start < i+1:
                sentences.append(text[start:i+1])
                start = i+1
            doc.sentences = sentences

    def bio_tag_generate_all_by_event(self):
        bio_tags = set()
        entity_bios = []
        # {span: count}
        dup_bio_tag_count = {}
        for doc in self.all_docs:
            # {span: bio}
            entity_bio = {}
            for event in doc.events:
                for k, v in event.items():
                    if k == 'EventType':
                        continue
                    if v in entity_bio:
                        if entity_bio[v] == 'trigger':
                            continue
                        if k == 'trigger':
                            continue
                        if entity_bio[v] != k or k not in entity_bio[v]:
                            if isinstance(entity_bio[v], list):
                                entity_bio[v].append(k)
                            else:
                                entity_bio[v] = [entity_bio[v], k]
                    else:
                        entity_bio[v] = k
                    bio_tags.add(k)
            entity_bios.append(entity_bio)

            for k, v in entity_bio.items():
                if isinstance(v, list):
                    v = sorted(v)
                    v = tuple(v)
                    if v in dup_bio_tag_count:
                        dup_bio_tag_count[v] += 1
                    else:
                        dup_bio_tag_count[v] = 1

        # {dup_list: span}
        dup_bio_map = {}
        for k in dup_bio_tag_count:
            v = dup_bio_tag_count[k]
            if v > 32:
                dup_bio_map[k] = '+'.join(list(k))
                bio_tags.add(dup_bio_map[k])
            else:
                dup_bio_map[k] = random.choice(k)

        for entity_bio in entity_bios:
            for k in entity_bio:
                v = entity_bio[k]
                if isinstance(v, list):
                    v = tuple(sorted(v))
                    entity_bio[k] = dup_bio_map[v]

        for doc, entity_bio in zip(self.all_docs, entity_bios):
            entities_example = []
            for entity, bio in entity_bio.items():
                positions = []
                for sentence_index in range(len(doc.sentences)):
                    sentence = doc.sentences[sentence_index]
                    start = 0
                    pos_starts = []
                    while True:
                        pos = sentence[start:].find(entity)
                        if pos == -1:
                            break
                        pos_starts.append(start+pos)
                        start += pos + len(entity)
                    position = [[sentence_index, x, x + len(entity)] for x in pos_starts]
                    positions += position
                entity_example = DuEEfinEntity(span=entity, positions=positions, field=bio)
                entities_example.append(entity_example)
            doc.entities = entities_example

    def get_pseudo_Doc2EDAG_processor(self):
        train_doc = self.all_docs[:self.train_num]
        dev_doc = self.all_docs[self.train_num:]
        assert len(train_doc) + len(dev_doc) == len(self.all_docs) and len(dev_doc) == self.dev_num

        new_train_dev_docs = []
        for docs in [train_doc, dev_doc]:
            new_docs = []
            for doc in docs:
                if len(doc.events) == 0:
                    continue
                new_entities = []
                for old_entity in doc.entities:
                    if old_entity.field == 'trigger':
                        continue
                    new_entity = DocEEEntity(span=old_entity.span, positions=old_entity.positions, field=old_entity.field)
                    new_entities.append(new_entity)
                new_events = []
                for old_event in doc.events:
                    new_event = {k: v for k, v in old_event.items() if k != 'trigger'}
                    new_events.append(new_event)
                new_doc = DocEEDocumentExample(doc_id=doc.doc_id, sentences=doc.sentences, events=new_events, entities=new_entities)
                new_docs.append(new_doc)
            new_train_dev_docs.append(new_docs)

        new_train_doc = new_train_dev_docs[0]
        new_dev_doc = new_train_dev_docs[1]

        train_docs = new_train_doc
        dev_docs = new_dev_doc
        test_docs = new_dev_doc

        dee_pro = DocEEProcessor()
        dee_pro.train_docs = train_docs
        dee_pro.dev_docs = dev_docs
        dee_pro.test_docs = test_docs
        dee_pro.SCHEMA = self.SCHEMA
        dee_pro.SCHEMA_KEY_ENG_CHN = None
        dee_pro.SCHEMA_KEY_CHN_ENG = None

        return dee_pro

    def generate_pseudo_Doc2EDAG_data(self):
        dee_pro = self.get_pseudo_Doc2EDAG_processor()
        schema = dee_pro.SCHEMA
        all_docs = [dee_pro.train_docs, dee_pro.dev_docs, dee_pro.test_docs]
        splits = ['train', 'dev', 'test']
        for docs, split in zip(all_docs, splits):
            lines = []
            for doc in docs:
                ann_valid_mspans = sorted([entity.span for entity in doc.entities])
                ann_mspan2dranges = {entity.span: entity.positions for entity in doc.entities}
                ann_mspan2guess_field = {entity.span: entity.field for entity in doc.entities}
                all_positions = []
                for entity in doc.entities:
                    all_positions += entity.positions
                ann_valid_dranges = sorted(all_positions, key=lambda x: x[0])
                recguid_eventname_eventdict_list = []
                for i, event in enumerate(doc.events):
                    # arguments = {k: v for k, v in event.items() if k != 'EventType' and k != 'trigger'}
                    arguments = {x: event[x] if x in event else None for x in schema[event['EventType']]}
                    event_data = [i, event['EventType'], arguments]
                    recguid_eventname_eventdict_list.append(event_data)
                data = {'sentences': doc.sentences,
                        'ann_valid_mspans': ann_valid_mspans,
                        'ann_valid_dranges': ann_valid_dranges,
                        'ann_mspan2dranges': ann_mspan2dranges,
                        'ann_mspan2guess_field': ann_mspan2guess_field,
                        'recguid_eventname_eventdict_list': recguid_eventname_eventdict_list,
                        }
                line = [doc.doc_id, data]
                lines.append(line)
            UtilData.write_json_file(GlobalConfigManager.get_pseudo_Doc2EDAG_path() / (split + '.json'), lines)
        logging.info('SCHEMA: {}'.format(self.SCHEMA))
