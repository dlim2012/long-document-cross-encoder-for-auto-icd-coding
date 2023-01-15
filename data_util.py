import re
import gensim
import torch
import os
from torch.utils.data import Dataset
from constant import DATA_DIR, MIMIC_2_DIR, MIMIC_3_DIR
import sys
import numpy as np
import csv
from collections import defaultdict
import warnings
import json, ujson
import random
import string

warnings.filterwarnings('ignore', category=FutureWarning)

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


def create_main_code(ind2c):
    mc = list(set([c.split('.')[0] for c in set(ind2c.values())]))
    mc.sort()
    ind2mc = {ind: mc for ind, mc in enumerate(mc)}
    mc2ind = {mc: ind for ind, mc in ind2mc.items()}
    return ind2mc, mc2ind

class MimicFullDataset(Dataset):
    def __init__(self,
                 config_name,
                 version,
                 mode,
                 model_name,
                 truncate_length,
                 label_truncate_length=32,
                 term_count=4,
                 sort_method='max',
                 sample_method='pos_neg_random',
                 n_ranks_ance=100,
                 n_samples=(1, 19),
                 return_tensors="np",
                 loss='cross_entropy',
                 switch_qd=False,
                chunk_notes=False
                 ):
        self.version = version
        self.mode = mode

        if version == 'mimic2':
            raise NotImplementedError
        if version in ['mimic3', 'mimic3-50']:
            if mode in ['train', 'dev', 'test']:
                self.path = os.path.join(MIMIC_3_DIR, f"{version}_{mode}.json")


        if version in ['mimic3']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_full.csv")
        if version in ['mimic3-50']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_50.csv")

        print(self.train_path)
        with open(self.path, "r") as f:
            self.df = ujson.load(f)


        self.truncate_length = truncate_length

        self.ind2c, _ = load_full_codes(self.train_path, version=version)

        # self.part_icd_codes = list(self.ind2c.values())
        self.c2ind = {c: ind for ind, c in self.ind2c.items()}
        self.code_count = len(self.ind2c)
        if mode == "train":
            print(f'Code count: {self.code_count}')

        self.ind2mc, self.mc2ind = create_main_code(self.ind2c)
        self.main_code_count = len(self.ind2mc)
        if mode == "train":
            print(f'Main code count: {self.main_code_count}')


        # Length
        self.len = len(self.df)

        # input parameters
        self.n_pos, self.n_neg = n_samples
        self.return_tensors = return_tensors
        self.sample_method = sample_method
        self.model_name = model_name
        self.label_truncate_length = label_truncate_length
        self.term_count = term_count
        self.sort_method = sort_method
        self.loss = loss
        self.n_ranks_ance = n_ranks_ance
        self.switch_qd = switch_qd

        self.chunk_notes = chunk_notes

        # Tokenizer
        from transformers import AutoConfig, AutoTokenizer
        self.config_name = config_name
        self.tokenizer = AutoTokenizer.from_pretrained(config_name)
        print('tokenizer config:', config_name)
        if self.config_name in ["yikuan8/Clinical-Longformer", "GanjinZero/biobart-v2-large"]:
            CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id = self.tokenizer.convert_tokens_to_ids(
                ['<s>', '</s>', '<mask>', '<pad>'])
            assert (CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id) == (0, 2, 50264, 1)
        elif self.config_name in ["RoBERTa-base-PM-M3-Voc-distill-align-hf", "RoBERTa-large-PM-M3-Voc-hf"]:
            CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id = self.tokenizer.convert_tokens_to_ids(
                ['<s>', '</s>', '<mask>', '<pad>'])
            assert (CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id) == (0, 2, 50000, 1)
        else:
            raise NotImplementedError()

        if self.model_name == 'colbert':
            print('add special tokens: [Q], [D]')
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[Q]', '[D]']})
            self.punctuation_token_ids = set(self.tokenizer.convert_tokens_to_ids([ch for ch in string.punctuation]))
            Q_token_id, D_token_id = self.tokenizer.convert_tokens_to_ids(['[Q]', '[D]'])
            self.token_ids = CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id, Q_token_id, D_token_id
        else:
            self.token_ids = CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id

        # list for note inputs
        self.note_inputs = [None] * self.len

        # Prepare token ids of code descriptions
        if self.mode == "train":
            self.prepare_label_feature(self.label_truncate_length)

        # labels
        self.labels = [[str(self.df[index]['LABELS']).split(';')] for index in range(self.len)]
        self.labels = [set([self.c2ind[label] for label in labels[0] if label in self.c2ind]) for labels in self.labels]

        # Prepare data structures depending on the sampling method
        self.n_ranks_ance = n_ranks_ance
        if self.sample_method == 'ance':
            self.labels_cands = [set(cands) for cands in self.labels]
            self.ranks = np.array([[self.c2ind[c] for c in line.split(';')[:self.n_ranks_ance] if c in self.c2ind] for line in top300])
        elif self.sample_method == 'pos_neg_top300':
            top300 = load_top300(os.path.join(DATA_DIR, 'top300', 'mimic3_%s_predtop50.txt' % mode))
            self.top300 = [set([self.c2ind[c] for c in line.split(';') if c in self.c2ind]) for line in top300]
            self.labels_cands = [set(cands) for cands in self.labels]
            self.neg_cands = [set() for cands in self.top300]
        elif self.sample_method in ['pos_neg_random', 'one', 'in-batch']:
            self.npr = self.n_neg / (self.n_pos + self.n_neg)
            self.labels_cands = [set(cands) for cands in self.labels]
            all_labels = set([label for label in self.c2ind.values()])
            self.neg_cands = [list(all_labels - cands) for cands in self.labels]
        else:
            raise ValueError()

        # binary labels
        if version == 'mimic3':
            self.binary_labels = np.zeros((self.len, 8921))
            for i in range(self.len):
                labels = list(self.labels[i])
                self.binary_labels[i][labels] = 1
        elif version == 'mimic3-50':
            self.binary_labels = np.zeros((self.len, 50))
            for i in range(self.len):
                labels = list(self.labels[i])
                self.binary_labels[i][labels] = 1

        if self.sample_method == 'in-batch':
            self.perm = np.arange(self.len)

    def __len__(self):
        if self.sample_method == 'in-batch':
            return self.len // self.n_pos
        return self.len

    def shuffle_indices(self):
        if self.mode == 'train':
            np.random.shuffle(self.perm)

    def gettext(self, index):
        return self.df[index]['TEXT']

    def split(self, text):
        sp = re.sub(r'\n\n+|  +', '\t', text.strip()).replace("\n",
                                                              " ").replace("!", "\t").replace("?", "\t").replace(".",
                                                                                                                 "\t")
        return [s.strip() for s in sp.split("\t") if s.strip()]


    def __getitem__(self, index):

        if self.sample_method == 'ance':
            note_inputs = self.get_note_inputs(index)
            negatives = self.sample_negatives(index, self.n_neg, method='ance')
            positives = self.sample_positives(index, self.n_pos, method='alternate')
            idc = positives + negatives
            labels = [1] * len(positives) + [0] * len(negatives)
        elif self.sample_method == 'pos_neg_top300':
            note_inputs = self.get_note_inputs(index)
            negatives = self.sample_negatives(index, self.n_neg, method='top300_alternate')
            positives = self.sample_positives(index, self.n_pos, method='alternate')
            idc = positives + negatives
            labels = [1] * len(positives) + [0] * len(negatives)
        elif self.sample_method == 'pos_neg_random':
            note_inputs = self.get_note_inputs(index)
            negatives = self.sample_negatives(index, self.n_neg, method='random')
            positives = self.sample_positives(index, self.n_pos, method='alternate')
            idc = positives + negatives
            labels = [1] * len(positives) + [0] * len(negatives)
        elif self.sample_method == 'one':
            note_inputs = self.get_note_inputs(index)
            if random.random() < self.npr or len(self.labels[index]) == 0:
                idc = self.sample_negatives(index, 1, method='random')
                labels = [0]
            else:
                idc = self.sample_positives(index, 1, method='alternate')
                labels = [1]
        elif self.sample_method == 'in-batch':
            note_indices, idc = self.sample_in_batch(self.perm[self.n_pos*index:self.n_pos*(index+1)])
            if len(note_indices) < 2:
                return -1
            note_inputs = [self.get_note_inputs(i, pad=True) for i in note_indices]
            note_inputs = {
                'input_ids': torch.cat([inputs['input_ids'] for inputs in note_inputs]),
                'attention_mask': torch.cat([inputs['attention_mask'] for inputs in note_inputs]),
                'filter_indices': torch.stack([inputs['filter_indices'] for inputs in note_inputs])
            }
            labels = [-1]
        else:
            raise ValueError()

        if self.loss == 'cross_entropy':
            if labels[0] == 0:  # there are no positive indices for 4 datapoints
                return -1
            else:
                labels = torch.tensor(0)
        elif self.loss == 'bce':
            labels = torch.tensor(labels, dtype=torch.float32)

        descs_inputs = {
            'input_ids': self.c_desc_input_ids[idc],
            'attention_mask': self.c_desc_attention_mask[idc]
        }

        if self.switch_qd:
            return descs_inputs, note_inputs, labels
        return note_inputs, descs_inputs, labels

    def extract_label_desc(self, ind2c):
        if not hasattr(self, 'desc_dict'):
            self.desc_dict = load_code_descriptions()

        desc_list = []
        for i in ind2c:
            code = ind2c[i]
            if not code in self.desc_dict:
                print(f'Not find desc of {code}')
            desc = self.desc_dict.get(code, code)
            desc_list.append(desc)
        return desc_list

    def process_label(self, ind2c, truncate_length, term_count=1, method='max'):
        desc_list = self.extract_label_desc(ind2c)
        if term_count == 1:
            c_desc_list = desc_list
        else:
            c_desc_list = []
            with open(f'../embedding/icd_mimic3_{method}_sort.json', 'r') as f:
                icd_syn = ujson.load(f)

            synonyms_dict = dict()
            for i in range(len(ind2c)):
                code = ind2c[i]
                new_terms = icd_syn.get(code, [])
                synonyms_dict[self.ind2c[i]] = [desc_list[i]] + new_terms
            self.synonyms_dict = synonyms_dict

            for i in sorted(ind2c):
                code = ind2c[i]
                tmp_desc = [desc_list[i]]
                new_terms = icd_syn.get(code, [])
                if len(new_terms) >= term_count - 1:
                    tmp_desc.extend(new_terms[0:term_count - 1])
                else:
                    tmp_desc.extend(new_terms)
                    repeat_count = int(term_count / len(tmp_desc)) + 1
                    tmp_desc = (tmp_desc * repeat_count)[0:term_count]
                if i < 5:
                    print(code, tmp_desc)

                c_desc_list.append(tmp_desc)

        self.c_desc_dict = []
        self.c_desc_input_ids = []
        self.c_desc_attention_mask = []
        self.c_desc_filter_indices = []
        if self.model_name == 'colbert':
            CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id, Q_token_id, D_token_id = self.token_ids

            for i, desc in enumerate(c_desc_list):
                desc = '. '.join(desc) + '.'
                token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(desc))[:self.label_truncate_length]
                if self.switch_qd: # c_desc is query
                    token_ids += [MASK_token_id] * (self.label_truncate_length - len(token_ids))
                    input_ids = [CLS_token_id, Q_token_id] + token_ids + [SEP_token_id]
                    attention_mask = [1] * (self.label_truncate_length + 3)
                else: # c_desc is document
                    input_ids = [CLS_token_id, D_token_id] + token_ids + [SEP_token_id]
                    attention_mask = [1] * (self.label_truncate_length + 3)
                    filter_indices = [i for i, input_id in enumerate(input_ids) if input_id not in self.punctuation_token_ids]
                    input_ids += [PAD_token_id] * (self.label_truncate_length- len(token_ids))
                    self.c_desc_filter_indices.append(torch.tensor(filter_indices))
                self.c_desc_dict.append(desc)
                self.c_desc_input_ids.append(torch.tensor(input_ids))
                self.c_desc_attention_mask.append(torch.tensor(attention_mask))
        else:
            for i, desc in enumerate(c_desc_list):
                desc = '. '.join(desc) + '.'
                inputs = self.tokenizer(desc, return_tensors=self.return_tensors, max_length=self.label_truncate_length,
                                        truncation=True, padding='max_length')
                self.c_desc_dict.append(desc)
                self.c_desc_input_ids.append(inputs['input_ids'][0])
                self.c_desc_attention_mask.append(inputs['attention_mask'][0])

        if self.return_tensors == 'pt':
            self.c_desc_input_ids = torch.stack(self.c_desc_input_ids)
            self.c_desc_attention_mask = torch.stack(self.c_desc_attention_mask)

    def prepare_label_feature(self, truncate_length):
        print('Prepare Label Feature')
        if hasattr(self, 'term_count'):
            term_count = self.term_count
        else:
            term_count = 1
        if hasattr(self, 'sort_method'):
            sort_method = self.sort_method
        else:
            sort_method = 'max'

        print('term_count', term_count)
        self.process_label(self.ind2c, truncate_length, term_count=term_count, method=sort_method)

    def tokenize_to_chunks(self, tokenizer, sentence, truncate_length, pad=True):

        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))[:truncate_length]
        n_chunks = (len(ids) - 1) // 509 + 1
        chunk_size = len(ids) // n_chunks + 1
        splits = np.arange(n_chunks + 1) * chunk_size

        input_ids_list = []
        attention_mask_list = []
        filter_indices_list = []
        if self.model_name == 'colbert':
            CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id, Q_token_id, D_token_id = self.token_ids

            for i in range(n_chunks):
                chunk = ids[splits[i]:splits[i + 1]]
                if self.switch_qd:
                    input_ids = [CLS_token_id + D_token_id] + chunk + [SEP_token_id]
                    attention_mask = [1] * len(input_ids)
                    if pad:
                        pad_len = chunk_size - len(chunk)
                        input_ids += [PAD_token_id] * pad_len
                        attention_mask += [0] * pad_len
                else:
                    input_ids = chunk + [MASK_token_id] * (chunk_size - len(chunk))
                    input_ids = [CLS_token_id, Q_token_id] + input_ids + [SEP_token_id]
                    attention_mask = [1] * len(input_ids)
                input_ids_list.append(torch.tensor(input_ids))
                attention_mask_list.append(torch.tensor(attention_mask))
        else:
            CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id = self.token_ids

            for i in range(n_chunks):
                chunk = ids[splits[i]:splits[i + 1]]
                if not pad:
                    input_ids = [CLS_token_id] + chunk + [SEP_token_id]
                    attention_mask = [1] * (2 + len(chunk))
                else:
                    pad_length = chunk_size - len(chunk)
                    input_ids = [CLS_token_id] + chunk + [SEP_token_id] + [PAD_token_id] * pad_length
                    attention_mask = [1] * (2 + len(chunk)) + [0] * pad_length
                input_ids_list.append(torch.tensor(input_ids))
                attention_mask_list.append(torch.tensor(attention_mask))
                if self.model_name == 'colbert':
                    filter_indices = [i for i, input_id in enumerate(input_ids) if input_id not in self.punctuation_token_ids]
                    filter_indices_list.append(filter_indices)
        return input_ids_list, attention_mask_list, filter_indices_list

    def get_note_inputs(self, index, pad=False):
        if self.note_inputs[index] == None:
            note = self.gettext(index)
            if self.chunk_notes:
                input_ids_list, attention_mask_list, filter_indices_list = self.tokenize_to_chunks(self.tokenizer, note, self.truncate_length, pad=True)
                inputs = {
                    'input_ids': torch.stack(input_ids_list),
                    'attention_mask': torch.stack(attention_mask_list)
                }
                if self.switch_qd:
                    inputs['filter_indices'] = torch.tensor(filter_indices_list)
            elif self.model_name == 'colbert':
                if self.config_name == "yikuan8/Clinical-Longformer":
                    CLS_token_id, SEP_token_id, MASK_token_id, PAD_token_id = self.tokenizer.convert_tokens_to_ids(['<s>', '</s>', '<mask>', '<pad>'])
                    Q_token_id, D_token_id = self.tokenizer.convert_tokens_to_ids(['[Q]', '[D]'])
                else:
                    raise ValueError("Set tokens for config:", self.config_name)

                inputs = self.tokenizer(note, max_length=self.truncate_length - 1, truncation=True)
                input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
                inputs = {}
                if self.switch_qd: # medical note is the query
                    input_ids = [input_ids[:1] + [D_token_id] + input_ids[1:]]
                    attention_mask = [attention_mask[:1] + [1] + attention_mask[1:]]
                    if pad:
                        pad_len = (self.truncate_length - len(input_ids[0]))
                        input_ids[0] += [PAD_token_id] * pad_len
                        attention_mask[0] += [0] * pad_len
                    filter_indices = [1 if input_id not in self.punctuation_token_ids else 0 for i, input_id in enumerate(input_ids[0])]
                    inputs['filter_indices'] = torch.tensor(filter_indices)
                else:
                    input_ids = input_ids[1:-1] + [MASK_token_id] * (self.truncate_length - len(input_ids))
                    input_ids = [[CLS_token_id, Q_token_id] + input_ids + [SEP_token_id]]
                    attention_mask = [[1] * len(input_ids[0])]

                inputs['input_ids'] = torch.tensor(input_ids)
                inputs['attention_mask'] = torch.tensor(attention_mask)

            else:
                if pad:
                    inputs = self.tokenizer(note, max_length=self.truncate_length, padding='max_length',
                                            return_tensors=self.return_tensors)
                else:
                    inputs = self.tokenizer(note, max_length=self.truncate_length, truncation=True,
                                            return_tensors=self.return_tensors)

            self.note_inputs[index] = inputs
        return self.note_inputs[index]

    def sample_negatives(self, index, n_neg=20, method='top300_random'):

        labels = set(self.labels[index])
        res = set()
        if method == 'random':
            if (len(self.neg_cands[index]) < n_neg): self.neg_cands[index] *= 2
            res = random.choices(self.neg_cands[index], k=n_neg)
        elif method == 'top300_random':
            cands = set(self.top300[index])
            cands = cands - labels
            res = random.sample(cands, n_neg)
        elif method == 'top300_alternate':
            if len(self.neg_cands[index]) < n_neg:
                to_add = set(self.neg_cands[index])
                self.neg_cands[index] = set(self.top300[index]) - labels - to_add
                res = set(random.sample(self.neg_cands[index], k=n_neg-len(to_add)))
                self.neg_cands[index] = (self.neg_cands[index] - res)
                res = res | to_add
            else:
                res = set(random.sample(self.neg_cands[index], n_neg))
                self.neg_cands[index] = self.neg_cands[index] - res
        elif method == 'ance':
            cands = list(set(self.ranks[index].tolist()) - self.labels[index])
            if len(cands) < n_neg:
                print('ance: len(cands) = %d, n_neg = %d' % (len(cands), n_neg))
            while len(cands) < n_neg:
                cands += cands
            res = random.sample(cands, n_neg)
        else:
            raise ValueError()

        return list(res)

    def sample_positives(self, index, n_pos, method='alternate'):
        res = set()
        if method == 'alternate':
            if len(self.labels_cands[index]) < n_pos:
                to_add = set(self.labels_cands[index])
                self.labels_cands[index] = set(self.labels[index]) - to_add
                try:
                    res = set(random.sample(self.labels_cands[index], k=n_pos-len(to_add)))
                except:
                    print(self.labels_cands[index], self.n_pos, to_add) # 4 training datapoints have 0 labels
                self.labels_cands[index] = (self.labels_cands[index] - res) | to_add
                res = res | to_add
            else:
                res = set(random.sample(self.labels_cands[index], n_pos))
                self.labels_cands[index] = self.labels_cands[index] - res
        else:
            raise ValueError()

        return list(res)

    def sample_in_batch(self, note_indices):
        res = [[], []] # [[note indices to use], [c_desc to use]]
        for index1 in note_indices:
            # choose positive
            exclude = set()
            for index2 in note_indices:
                if index1 == index2:
                    continue
                exclude = exclude.union(self.labels[index2])
            cands = [i for i in self.labels_cands[index1] if i not in exclude]
            if len(cands) == 0:
                cands = [i for i in self.labels[index1] if i not in exclude]
            if len(cands) == 0:
                continue
            pos = random.choice(cands)
            if pos in self.labels_cands[index1]:
                self.labels_cands[index1].remove(pos)
            res[0].append(index1)
            res[1].append(pos)
        return res



def my_collate_fn(batch):
    type_count = len(batch[0])
    batch_size = len(batch)
    output = ()
    for i in range(type_count):
        tmp = []
        for item in batch:
            tmp.extend(item[i])
        if len(tmp) <= batch_size:
            output += (torch.LongTensor(tmp),)
        elif isinstance(tmp[0], int):
            output += (torch.LongTensor(tmp).reshape(batch_size, -1),)
        elif isinstance(tmp[0], float):
            output += (torch.FloatTensor(tmp).reshape(batch_size, -1),)
        elif isinstance(tmp[0], list):
            dim_y = len(tmp[0])
            if isinstance(tmp[0][0], int):
                output += (torch.LongTensor(tmp).reshape(batch_size, -1, dim_y),)
            elif isinstance(tmp[0][0], float):
                output += (torch.FloatTensor(tmp).reshape(batch_size, -1, dim_y),)
    return output


def load_vocab(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        words = [line.strip().split()[0] for line in lines]
    except BaseException:
        if path.endswith('.model'):
            model = gensim.models.Word2Vec.load(path)
        if path.endswith('.bin'):
            model = gensim.models.KeyedVectors.load_word2vec_format(
                path, binary=True)
        words = list(model.wv.vocab)
        del model

    # hard code to trim word embedding size
    try:
        with open('./embedding/word_count_dict.json', 'r') as f:
            word_count_dict = ujson.load(f)
    except BaseException:
        with open('../embedding/word_count_dict.json', 'r') as f:
            word_count_dict = ujson.load(f)
    words = [w for w in words if w in word_count_dict]

    for w in ["**UNK**", "**PAD**", "**MASK**"]:
        if not w in words:
            words = words + [w]
    word2id = {word: idx for idx, word in enumerate(words)}
    id2word = {idx: word for idx, word in enumerate(words)}
    return word2id, id2word


def load_full_codes(train_path, version='mimic3'):
    """
        Inputs:
            train_path: path to train dataset
            version: which (MIMIC) dataset
        Outputs:
            code lookup, description lookup
    """
    # get description lookup
    desc_dict = load_code_descriptions(version=version)
    # build code lookups from appropriate datasets
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open('%s/proc_dsums.csv' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            # header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    return ind2c, desc_dict


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def load_code_descriptions(version='mimic3'):
    # load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    if version == 'mimic2':
        with open('%s/MIMIC_ICD9_mapping' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            # header
            next(r)
            for row in r:
                desc_dict[str(row[1])] = str(row[2])
    else:
        with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            # header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                desc_dict[reformat(code, True)] = desc
        with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            # header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                if code not in desc_dict.keys():
                    desc_dict[reformat(code, False)] = desc
        with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
            for _, row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = ' '.join(row[1:])
    return desc_dict


def load_embeddings(embed_file):
    W = []
    word_list = []
    try:
        with open(embed_file) as ef:
            for line in ef:
                line = line.rstrip().split()
                word_list.append(line[0])
                vec = np.array(line[1:]).astype(np.float)
                # also normalizes the embeddings
                vec = vec / float(np.linalg.norm(vec) + 1e-6)
                W.append(vec)
        word2id, id2word = load_vocab(embed_file)
    except BaseException:
        if embed_file.endswith('.model'):
            model = gensim.models.Word2Vec.load(embed_file)
        if embed_file.endswith('.bin'):
            model = gensim.models.KeyedVectors.load_word2vec_format(
                embed_file, binary=True)
        words = list(model.wv.vocab)

        original_word_count = len(words)

        # hard code to trim word embedding size
        with open('./embedding/word_count_dict.json', 'r') as f:
            word_count_dict = ujson.load(f)
        words = [w for w in words if w in word_count_dict]

        for w in ["**UNK**", "**PAD**", "**MASK**"]:
            if not w in words:
                words = words + [w]
        word2id = {word: idx for idx, word in enumerate(words)}
        id2word = {idx: word for idx, word in enumerate(words)}
        new_W = []
        for i in range(len(id2word)):
            if not id2word[i] in ["**UNK**", "**PAD**", "**MASK**"]:
                new_W.append(model.__getitem__(id2word[i]))
            elif id2word[i] == "**UNK**":
                print("adding unk embedding")
                new_W.append(np.random.randn(len(new_W[-1])))
            elif id2word[i] == "**MASK**":
                print("adding mask embedding")
                new_W.append(np.random.randn(len(new_W[-1])))
            elif id2word[i] == "**PAD**":
                print("adding pad embedding")
                new_W.append(np.zeros_like(new_W[-1]))
        new_W = np.array(new_W)
        print(f"Word count: {len(id2word)}")
        print(f"Load embedding count: {len(new_W)}")
        print(
            f"Original word count: {original_word_count}/{len(word_count_dict)}")
        del model
        return new_W

    if not "**UNK**" in word_list:
        # UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        word_list.append("**UNK**")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    if not "**MASK**" in word_list:
        # UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        word_list.append("**UNK**")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    if not "**PAD**" in word_list:
        print("adding pad embedding")
        word_list.append("**PAD**")
        vec = np.zeros_like(W[-1])
        W.append(vec)

    print(f"Word count: {len(id2word)}")
    print(f"Load embedding count: {len(W)}")
    print(f"Original word count: {original_word_count}/{len(word_count_dict)}")
    word2newid = {w: i for i, w in enumerate(word_list)}
    new_W = []
    for i in range(len(id2word)):
        new_W.append(W[word2newid[id2word[i]]])
    new_W = np.array(new_W)
    del model
    return new_W

def load_top300(file):
    with open(file, 'r') as f:
        top300 = f.read().splitlines()
    return top300

