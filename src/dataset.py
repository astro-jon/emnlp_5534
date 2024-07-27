import re
import torch
from torch.utils.data import Dataset
from Untitled import *
from cfg import CFG
import numpy as np


class InputDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = eval(self.data['sentence_mask'][item])
        options = eval(self.data['options'][item])
        dis = [j for i in options for j in i]
        opt_len = np.sum(eval(self.data['opt_len'][item]))
        options = [options]
        answer = [self.data['answer'][item]]
        position = [self.data['opt_pos'][item]]
        true_input_len = len(text) + 2

        # article input
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]']) +
                                 [0] * (256 - true_input_len))
        attention_mask = torch.tensor([1] * true_input_len + [0] * (256 - true_input_len))
        token_type_ids = torch.tensor([0] * true_input_len + [0] * (256 - true_input_len))
        # input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]'] + dis +
        #                                                               ['[SEP]']) + [0] * (1024 - true_input_len))
        # attention_mask = torch.tensor([1] * true_input_len + [0] * (1024 - true_input_len))
        # token_type_ids = torch.tensor([0] * (len(text) + 2) + [1] * (opt_len + 1) + [0] * (1024 - true_input_len))
        # label = torch.tensor(self.data['level'][item], dtype=torch.long)

        # option
        options_input = torch.zeros(CFG.max_option_num, 4, CFG.max_option_len).long()
        options_mask = torch.ones(options_input.size())
        for i, opt in enumerate(options):
            for k, op in enumerate(opt):
                op_embedding = torch.tensor(self.tokenizer.convert_tokens_to_ids(op))
                options_input[i, k, :op_embedding.size(0)] = op_embedding
                options_mask[i, k, op_embedding.size(0):] = 0

        # answer
        answers_input = torch.zeros(CFG.max_option_num).long()
        answers_mask = torch.zeros(answers_input.size())
        labels = torch.full(answers_input.size(), -100)
        for i, ans in enumerate(answer):
            ans_embedding = torch.tensor(ord(ans)-ord('A'))
            answers_input[i] = ans_embedding
            labels[i] = ans_embedding
            answers_mask[i] = 1

        # MASK position
        position_input = torch.zeros(answers_input.size()).long()
        for i, pos in enumerate(position):
            pos_embedding = torch.tensor(pos)
            position_input[i] = pos_embedding

        return {
            'text': ' '.join(text),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'options_input': options_input,
            'options_mask': options_mask,
            'answers_input': answers_input,
            'answers_mask': answers_mask,
            'position_input': position_input,
            'label': labels,
            # 'level': level
        }


class InputDatasetOption(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_id = self.data['item_id'][item]
        text = eval(self.data['sentence_mask'][item])
        options = eval(self.data['options'][item])
        dis = [j for i in options for j in i]
        opt_len = np.sum(eval(self.data['opt_len'][item]))
        options = [options]
        answer = [self.data['answer'][item]]
        position = [self.data['opt_pos'][item]]
        # level = self.data['level'][item]
        true_input_len = len(text) + opt_len + 3

        # article input
        # input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + dis + ['[SEP]']) +
        #                          [0] * (1024 - true_input_len))
        # attention_mask = torch.tensor([1] * true_input_len + [0] * (1024 - true_input_len))
        # token_type_ids = torch.tensor([0] * 1024)
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]'] + dis +
                                                                      ['[SEP]']) + [0] * (CFG.max_input_length - true_input_len))
        attention_mask = torch.tensor([1] * true_input_len + [0] * (CFG.max_input_length - true_input_len))
        token_type_ids = torch.tensor([0] * (len(text)+2) + [1] * (opt_len+1) + [0] * (CFG.max_input_length-true_input_len))

        # option
        options_input = torch.zeros(CFG.max_option_num, 4, CFG.max_option_len).long()
        options_mask = torch.ones(options_input.size())
        for i, opt in enumerate(options):
            for k, op in enumerate(opt):
                op_embedding = torch.tensor(self.tokenizer.convert_tokens_to_ids(op))
                options_input[i, k, :op_embedding.size(0)] = op_embedding
                options_mask[i, k, op_embedding.size(0):] = 0

        # answer
        answers_input = torch.zeros(CFG.max_option_num).long()
        answers_mask = torch.zeros(answers_input.size())
        labels = torch.full(answers_input.size(), -100)
        for i, ans in enumerate(answer):
            ans_embedding = torch.tensor(ord(ans)-ord('A'))
            answers_input[i] = ans_embedding
            labels[i] = ans_embedding
            answers_mask[i] = 1

        # MASK position
        position_input = torch.zeros(answers_input.size()).long()
        for i, pos in enumerate(position):
            pos_embedding = torch.tensor(pos)
            position_input[i] = pos_embedding

        return {
            'item_id': item_id,
            'text': ' '.join(text),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'options_input': options_input,
            'options_mask': options_mask,
            'answers_input': answers_input,
            'answers_mask': answers_mask,
            'position_input': position_input,
            'label': labels,
            # 'level': level
        }


class InputDatasetOptionFeature(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_id = self.data['item_id'][item]
        text = eval(self.data['sentence_mask'][item])
        options = eval(self.data['options'][item])
        dis = [j for i in options for j in i]
        opt_len = np.sum(eval(self.data['opt_len'][item]))
        options = [options]
        answer = [self.data['answer'][item]]
        position = [self.data['opt_pos'][item]]
        orl_option = self.data['orl_options'][item]
        orl_sentence = self.data['orl_sentence'][item]
        answer_word = self.data['answer_word'][item]

        true_input_len = len(text) + opt_len + 3
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]'] + dis +
                                                                      ['[SEP]']) + [0] * (
                                             CFG.max_input_length - true_input_len))
        attention_mask = torch.tensor([1] * true_input_len + [0] * (CFG.max_input_length - true_input_len))
        token_type_ids = torch.tensor(
            [0] * (len(text) + 2) + [1] * (opt_len + 1) + [0] * (CFG.max_input_length - true_input_len))

        # option
        options_input = torch.zeros(CFG.max_option_num, 4, CFG.max_option_len).long()
        options_mask = torch.ones(options_input.size())
        for i, opt in enumerate(options):
            for k, op in enumerate(opt):
                op_embedding = torch.tensor(self.tokenizer.convert_tokens_to_ids(op))
                options_input[i, k, :op_embedding.size(0)] = op_embedding
                options_mask[i, k, op_embedding.size(0):] = 0

        # answer
        answers_input = torch.zeros(CFG.max_option_num).long()
        answers_mask = torch.zeros(answers_input.size())
        labels = torch.full(answers_input.size(), -100)
        for i, ans in enumerate(answer):
            ans_embedding = torch.tensor(ord(ans) - ord('A'))
            answers_input[i] = ans_embedding
            labels[i] = ans_embedding
            answers_mask[i] = 1

        # MASK position
        position_input = torch.zeros(answers_input.size()).long()
        for i, pos in enumerate(position):
            pos_embedding = torch.tensor(pos)
            position_input[i] = pos_embedding

        return {
            'item_id': item_id,
            'text': ' '.join(text),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'options_input': options_input,
            'options_mask': options_mask,
            'answers_input': answers_input,
            'answers_mask': answers_mask,
            'position_input': position_input,
            'label': labels,
            'orl_option': orl_option,
            'orl_sentence': orl_sentence,
            'answer_word': answer_word
        }


class InputDatasetWithVocab(Dataset):
    def __init__(self, data, tokenizer, vocab, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = eval(self.data['article_mask'][item])
        options = eval(self.data['options'][item])
        answer = eval(self.data['answers'][item])
        position = eval(self.data['opt_pos'][item])
        true_input_len = len(text) + 2
        opt_len = torch.tensor(len(position))

        # article input
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]']) +
                                 [0] * (1024 - true_input_len))
        attention_mask = torch.tensor([1] * true_input_len + [0] * (1024 - true_input_len))
        token_type_ids = torch.tensor([0] * true_input_len + [0] * (1024 - true_input_len))

        # options
        options_input = torch.zeros(CFG.max_option_num, 4).long()
        for i, option in enumerate(options):
            for j, opt in enumerate(option):
                options_input[i, j] = torch.tensor(self.vocab[opt])

        # answer
        answers_input = torch.zeros(CFG.max_option_num).long()
        answers_mask = torch.zeros(answers_input.size())
        for i, ans in enumerate(answer):
            ans_embedding = torch.tensor(ord(ans)-ord('A'))
            answers_input[i] = ans_embedding
            answers_mask[i] = 1

        # MASK position
        position_input = torch.zeros(answers_input.size()).long()
        for i, pos in enumerate(position):
            pos_embedding = torch.tensor(pos)
            position_input[i] = pos_embedding

        return {
            'text': ' '.join(text),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'options_input': options_input,
            'position_input': position_input,
            'answers_mask': answers_mask,
            'label': answers_input,
            'opt_len': opt_len
        }


class InputDataset4sentence(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = eval(self.data['sentence_mask'][item])
        options = eval(self.data['options'][item])
        position = [self.data['opt_pos'][item]]
        answer = self.data['answer'][item]
        answer_word = options[ord(self.data['answer'][item]) - ord('A')]
        opt_sent = []
        text_ans = None
        for opts in options:
            if opts == answer_word:
                opt = ' '.join(opts)
                text_ans = ' '.join(text).replace('[MASK]', opt).split() + ['[SEP]']
            else:
                opt = ' '.join(opts)
                text_dis = ' '.join(text).replace('[MASK]', opt).split() + ['[SEP]']
                opt_sent += text_dis
        dis = [j for i in options for j in i]
        opt_len = np.sum(eval(self.data['opt_len'][item]))
        options = [options]
        # answer = eval(self.data['answers'][item])
        # position = [self.data['opt_pos'][item]]
        true_input_len = len(text) + len(text_ans) + len(opt_sent) + 2

        # article input
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]'] + text_ans +
                                                                      opt_sent) + [0] * (1024 - true_input_len))
        attention_mask = torch.tensor([1] * true_input_len + [0] * (1024 - true_input_len))
        token_type_ids = torch.tensor([0] * (len(text)+2) + [1] * (len(text_ans)+len(opt_sent)) +
                                      [0] * (1024-true_input_len))

        # option
        options_input = torch.zeros(CFG.max_option_num, 4, CFG.max_option_len).long()
        options_mask = torch.ones(options_input.size())
        for i, opt in enumerate(options):
            for k, op in enumerate(opt):
                op_embedding = torch.tensor(self.tokenizer.convert_tokens_to_ids(op))
                options_input[i, k, :op_embedding.size(0)] = op_embedding
                options_mask[i, k, op_embedding.size(0):] = 0

        # answer
        answers_input = torch.zeros(CFG.max_option_num).long()
        answers_mask = torch.zeros(answers_input.size())
        labels = torch.full(answers_input.size(), -100)
        for i, ans in enumerate(answer):
            ans_embedding = torch.tensor(ord(ans)-ord('A'))
            answers_input[i] = ans_embedding
            labels[i] = ans_embedding
            answers_mask[i] = 1

        # MASK position
        position_input = torch.zeros(answers_input.size()).long()
        for i, pos in enumerate(position):
            pos_embedding = torch.tensor(pos)
            position_input[i] = pos_embedding

        return {
            'text': ' '.join(text),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'options_input': options_input,
            'options_mask': options_mask,
            'answers_input': answers_input,
            'answers_mask': answers_mask,
            'position_input': position_input,
            'label': labels,
        }


class InputDataset1sentence(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = eval(self.data['sentence_mask'][item])
        options = eval(self.data['options'][item])
        position = [self.data['opt_pos'][item]]
        answer = self.data['answer'][item]
        answer_word = options[ord(self.data['answer'][item]) - ord('A')]
        opt_sent = []
        text_ans = None
        for opts in options:
            if opts == answer_word:
                opt = ' '.join(opts)
                text_ans = ' '.join(text).replace('[MASK]', opt).split() + ['[SEP]']
            else:
                opt = ' '.join(opts)
                text_dis = ' '.join(text).replace('[MASK]', opt).split() + ['[SEP]']
                opt_sent += text_dis
        dis = [j for i in options for j in i]
        opt_len = np.sum(eval(self.data['opt_len'][item]))
        options = [options]
        # answer = eval(self.data['answers'][item])
        # position = [self.data['opt_pos'][item]]
        true_input_len = len(text) + len(text_ans) + 2

        # article input
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]'] + text_ans) +
                                 [0] * (1024 - true_input_len))
        attention_mask = torch.tensor([1] * true_input_len + [0] * (1024 - true_input_len))
        token_type_ids = torch.tensor([0] * (len(text)+2) + [1] * len(text_ans) +
                                      [0] * (1024-true_input_len))

        # option
        options_input = torch.zeros(CFG.max_option_num, 4, CFG.max_option_len).long()
        options_mask = torch.ones(options_input.size())
        for i, opt in enumerate(options):
            for k, op in enumerate(opt):
                op_embedding = torch.tensor(self.tokenizer.convert_tokens_to_ids(op))
                options_input[i, k, :op_embedding.size(0)] = op_embedding
                options_mask[i, k, op_embedding.size(0):] = 0

        # answer
        answers_input = torch.zeros(CFG.max_option_num).long()
        answers_mask = torch.zeros(answers_input.size())
        labels = torch.full(answers_input.size(), -100)
        for i, ans in enumerate(answer):
            ans_embedding = torch.tensor(ord(ans)-ord('A'))
            answers_input[i] = ans_embedding
            labels[i] = ans_embedding
            answers_mask[i] = 1

        # MASK position
        position_input = torch.zeros(answers_input.size()).long()
        for i, pos in enumerate(position):
            pos_embedding = torch.tensor(pos)
            position_input[i] = pos_embedding

        return {
            'text': ' '.join(text),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'options_input': options_input,
            'options_mask': options_mask,
            'answers_input': answers_input,
            'answers_mask': answers_mask,
            'position_input': position_input,
            'label': labels,
        }


class InputDatasetPromptBefore(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = eval(self.data['sentence_mask'][item])
        options = eval(self.data['options'][item])
        sentence = 'Which option fits the blank : {} , {} , {} , {} .'
        sentence = (' '.join(self.tokenizer.tokenize(sentence)).replace('▁{}', '{}').
                    format(' '.join(options[0]), ' '.join(options[1]),
                           ' '.join(options[2]), ' '.join(options[3]))).split()
        sentence_len = len(sentence)
        options = [options]
        answer = [self.data['answer'][item]]
        position = [self.data['opt_pos'][item]+sentence_len]
        true_input_len = len(text) + len(sentence) + 2

        # article input
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence + text + ['[SEP]']) + [0] * (CFG.max_input_length - true_input_len))
        # input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence + ['[SEP]'] + text + ['[SEP]']) + [0] * (CFG.max_input_length - true_input_len))
        attention_mask = torch.tensor([1] * true_input_len + [0] * (CFG.max_input_length - true_input_len))
        # token_type_ids = torch.tensor([0] * (len(text) + 2) + [1] * (len(sentence) + 1) + [0] * (CFG.max_input_length - true_input_len))
        # token_type_ids = torch.tensor([0] * (len(sentence)+2) + [1] * (len(text)+1) + [0] * (CFG.max_input_length-true_input_len))
        token_type_ids = torch.tensor([0] * CFG.max_input_length)

        # option
        options_input = torch.zeros(CFG.max_option_num, 4, CFG.max_option_len).long()
        options_mask = torch.ones(options_input.size())
        for i, opt in enumerate(options):
            for k, op in enumerate(opt):
                op_embedding = torch.tensor(self.tokenizer.convert_tokens_to_ids(op))
                options_input[i, k, :op_embedding.size(0)] = op_embedding
                options_mask[i, k, op_embedding.size(0):] = 0

        # answer
        answers_input = torch.zeros(CFG.max_option_num).long()
        answers_mask = torch.zeros(answers_input.size())
        labels = torch.full(answers_input.size(), -100)
        for i, ans in enumerate(answer):
            ans_embedding = torch.tensor(ord(ans) - ord('A'))
            answers_input[i] = ans_embedding
            labels[i] = ans_embedding
            answers_mask[i] = 1

        # MASK position
        position_input = torch.zeros(answers_input.size()).long()
        for i, pos in enumerate(position):
            pos_embedding = torch.tensor(pos)
            position_input[i] = pos_embedding

        return {
            'text': ' '.join(text),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'options_input': options_input,
            'options_mask': options_mask,
            'answers_input': answers_input,
            'answers_mask': answers_mask,
            'position_input': position_input,
            'label': labels,
        }


class InputDatasetPromptAfter(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = eval(self.data['sentence_mask'][item])
        options = eval(self.data['options'][item])
        sentence = 'Which option fits the blank : {} , {} , {} , {} .'
        sentence = (' '.join(self.tokenizer.tokenize(sentence)).replace('▁{}', '{}').
                    format(' '.join(options[0]), ' '.join(options[1]),
                           ' '.join(options[2]), ' '.join(options[3]))).split()
        sentence_len = len(sentence)
        options = [options]
        answer = [self.data['answer'][item]]
        position = [self.data['opt_pos'][item]]
        true_input_len = len(text) + len(sentence) + 2

        # article input
        # input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]'] + sentence + ['[SEP]']) + [0] * (CFG.max_input_length - true_input_len))
        input_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text + sentence + ['[SEP]']) + [0] * (
                        CFG.max_input_length - true_input_len))
        attention_mask = torch.tensor([1] * true_input_len + [0] * (CFG.max_input_length - true_input_len))
        # token_type_ids = torch.tensor([0] * (len(text) + 2) + [1] * (len(sentence) + 1) + [0] * (CFG.max_input_length - true_input_len))
        # token_type_ids = torch.tensor([0] * (len(sentence)+2) + [1] * (len(text)+1) + [0] * (CFG.max_input_length-true_input_len))
        token_type_ids = torch.tensor([0] * CFG.max_input_length)

        # option
        options_input = torch.zeros(CFG.max_option_num, 4, CFG.max_option_len).long()
        options_mask = torch.ones(options_input.size())
        for i, opt in enumerate(options):
            for k, op in enumerate(opt):
                op_embedding = torch.tensor(self.tokenizer.convert_tokens_to_ids(op))
                options_input[i, k, :op_embedding.size(0)] = op_embedding
                options_mask[i, k, op_embedding.size(0):] = 0

        # answer
        answers_input = torch.zeros(CFG.max_option_num).long()
        answers_mask = torch.zeros(answers_input.size())
        labels = torch.full(answers_input.size(), -100)
        for i, ans in enumerate(answer):
            ans_embedding = torch.tensor(ord(ans) - ord('A'))
            answers_input[i] = ans_embedding
            labels[i] = ans_embedding
            answers_mask[i] = 1

        # MASK position
        position_input = torch.zeros(answers_input.size()).long()
        for i, pos in enumerate(position):
            pos_embedding = torch.tensor(pos)
            position_input[i] = pos_embedding

        return {
            'text': ' '.join(text),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'options_input': options_input,
            'options_mask': options_mask,
            'answers_input': answers_input,
            'answers_mask': answers_mask,
            'position_input': position_input,
            'label': labels,
        }






