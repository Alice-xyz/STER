import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import Iterable, List

from tqdm import tqdm
from transformers import BertTokenizer

from spert import util
from spert.entities import Dataset, EntityType, RelationType, Entity, Relation, Document
import numpy as np

class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, k: int = None, logger: Logger = None):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size
        self._k = k

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _calc_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, k: int = None,logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, k, logger)

    def read(self, dataset_paths):
        # dataset labels
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                              self._neg_rel_count, self._max_span_size, self._k)
            self._parse_dataset(dataset_path, dataset)
            self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']  # each sentence in dataset
        jrelations = doc['relations']
        jentities = doc['entities']

        # parse tokens
        doc_tokens, doc_encoding = self._parse_tokens(jtokens, dataset)

        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)  # 1 dim list

        # parse relations
        relations = self._parse_relations(jrelations, entities, dataset)  # 1 dim list

        # create document
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)

        return document

    # def _parse_tokens_TeaE(self, jtokens, dataset):
    #     doc_tokens = []
    #
    #     # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    #     # while doc_tokens does not include special tokens
    #     doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
    #
    #     # parse tokens
    #     for i, token_phrase in enumerate(jtokens):
    #         token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
    #         span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
    #
    #         token = dataset.create_token(i, span_start, span_end, token_phrase)
    #
    #         doc_tokens.append(token)
    #         doc_encoding += token_encoding
    #
    #     doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
    #
    #     return doc_tokens, doc_encoding

    def _parse_tokens(self, jtokens, dataset):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        # while doc_tokens does not include special tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []
        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)

            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        # for each document
        relations = []
        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)
        return relations


class JsonInputReaderTeaE(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, k: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, k, logger)

    def read(self, dataset_paths):
        # dataset labels
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                              self._neg_rel_count, self._max_span_size, self._k)
            self._parse_dataset(dataset_path, dataset)
            self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    # def _parse_document(self, doc, dataset) -> Document:
    #     jtokens = doc['tokens']  # each sentence in dataset
    #     jrelations = doc['relations']
    #     jentities = doc['entities']
    #
    #     # parse tokens
    #     doc_tokens, doc_encoding = self._parse_tokens(jtokens, dataset)
    #
    #     # parse entity mentions
    #     entities = self._parse_entities(jentities, doc_tokens, dataset)  # 1 dim list
    #     L_entities = np.array(entities)
    #     print("input_reader.py # L_entities.shape:%s, " % L_entities.shape)
    #
    #     # parse relations
    #     relations = self._parse_relations(jrelations, entities, dataset)  # 1 dim list
    #     L_relations = np.array(relations)
    #     print("input_reader.py # L_relations.shape:%s, " % L_relations.shape)
    #
    #     # create document
    #     document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)
    #
    #     print("input_reader.py # document.doc_id:%s, document.entities:, document.relations:, document.tokens:, document.encoding:%s, \n" % (document.doc_id, document.encoding) )
    #     return document

    # def _parse_document_teaE(self, doc, dataset) -> Document:
    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']  # each sentence in dataset
        jrelations = doc['relations']
        jentities = doc['entities']

        # parse tokens
        doc_tokens, doc_encoding = self._parse_tokens(jtokens, dataset)
        doc_encoding = doc_encoding[:-1]  # [SEP]

        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)  # 1 dim list
        # for entity_i in range(len(entities)):
        #     print("input_reader.py # entities[entity_i].phrase:%s, " % (entities[entity_i].phrase))  # entities[entity_i] object
        #     # print("input_reader.py # entities[entity_i].tokens.get_span:%s, " % (entities[entity_i].phrase.split()))  # entities[entity_i] object
        #     doc_encoding += self._tokenizer.encode(entities[entity_i].phrase)
        #     # doc_encoding += self._tokenizer.convert_tokens_to_ids(entities[entity_i].phrase.split())
        #
        # doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]  # add [SEP] again
        for entity in entities:
            add_tokens = []
            for token in entity.tokens:
                add_tokens += doc_encoding[token.span_start: token.span_end]
            doc_encoding += add_tokens
        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]  # add [SEP] again

        # parse relations
        relations = self._parse_relations(jrelations, entities, dataset)  # 1 dim list

        # create document
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)

        return document

    # def _parse_tokens_TeaE(self, jtokens, dataset):
    #     doc_tokens = []
    #
    #     # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    #     # while doc_tokens does not include special tokens
    #     doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
    #
    #     # parse tokens
    #     for i, token_phrase in enumerate(jtokens):
    #         token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
    #         span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
    #
    #         token = dataset.create_token(i, span_start, span_end, token_phrase)
    #
    #         doc_tokens.append(token)
    #         doc_encoding += token_encoding
    #
    #     doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
    #
    #     return doc_tokens, doc_encoding

    def _parse_tokens(self, jtokens, dataset):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        # while doc_tokens does not include special tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
        test_1 = "hellloo hi"
        test_1_encoding = self._tokenizer.encode(test_1, add_special_tokens=False)
        test_2 = ["hellloo", "hi"]
        test_2_encoding = self._tokenizer.encode(test_2, add_special_tokens=False)
        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []
        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)

            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        # for each document
        relations = []
        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)
        return relations


class JsonInputReaderTeaR(BaseInputReader):
# class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, k: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, k, logger)

    def read(self, dataset_paths):
        # dataset labels
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                              self._neg_rel_count, self._max_span_size, self._k)
            self._parse_dataset(dataset_path, dataset)
            self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    # def _parse_document(self, doc, dataset) -> Document:
    #     jtokens = doc['tokens']  # each sentence in dataset
    #     jrelations = doc['relations']
    #     jentities = doc['entities']
    #
    #     # parse tokens
    #     doc_tokens, doc_encoding = self._parse_tokens(jtokens, dataset)
    #
    #     # parse entity mentions
    #     entities = self._parse_entities(jentities, doc_tokens, dataset)  # 1 dim list
    #     L_entities = np.array(entities)
    #     print("input_reader.py # L_entities.shape:%s, " % L_entities.shape)
    #
    #     # parse relations
    #     relations = self._parse_relations(jrelations, entities, dataset)  # 1 dim list
    #     L_relations = np.array(relations)
    #     print("input_reader.py # L_relations.shape:%s, " % L_relations.shape)
    #
    #     # create document
    #     document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)
    #
    #     print("input_reader.py # document.doc_id:%s, document.entities:, document.relations:, document.tokens:, document.encoding:%s, \n" % (document.doc_id, document.encoding) )
    #     return document

    # def _parse_document_teaE(self, doc, dataset) -> Document:
    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']  # each sentence in dataset
        jrelations = doc['relations']
        jentities = doc['entities']

        # parse tokens
        doc_tokens, doc_encoding = self._parse_tokens(jtokens, dataset)
        doc_encoding = doc_encoding[:-1]  # [SEP]
        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)  # 1 dim list


        # parse relations
        relations = self._parse_relations(jrelations, entities, dataset)  # 1 dim list

        for relation in relations:
            doc_encoding += [relation.relation_type.index]
        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]  # add [SEP] again

        # create document
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)

        return document

    # def _parse_tokens_TeaE(self, jtokens, dataset):
    #     doc_tokens = []
    #
    #     # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    #     # while doc_tokens does not include special tokens
    #     doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
    #
    #     # parse tokens
    #     for i, token_phrase in enumerate(jtokens):
    #         token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
    #         span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
    #
    #         token = dataset.create_token(i, span_start, span_end, token_phrase)
    #
    #         doc_tokens.append(token)
    #         doc_encoding += token_encoding
    #
    #     doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
    #
    #     return doc_tokens, doc_encoding

    def _parse_tokens(self, jtokens, dataset):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        # while doc_tokens does not include special tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []
        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        # for each document
        relations = []
        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)
        return relations

# L = np.array(relations)
# print("input_reader.py # L.shape:%s, " % L.shape)

# for i in range(len(entity.tokens)):
#     print("input_reader.py # entity.tokens[%s]:%s, " % (i, entity.tokens[i].span))
