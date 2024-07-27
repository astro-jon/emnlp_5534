import torch
from transformers import BigBirdForMaskedLM, BigBirdModel, pipeline, AutoTokenizer
import torch.nn.functional as F
from torch import nn
from cfg import CFG


class BigBirdForSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = CFG.num_labels
        self.bigbird = BigBirdForMaskedLM.from_pretrained(CFG.model_path)

        self.bert_head = nn.Sequential(
            nn.Linear(self.bigbird.config.hidden_size, self.bigbird.config.vocab_size),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, options_input, options_mask, answers_input,
                answers_mask, position_input, labels=None):
        loss = None
        total_acc = None
        batch = min(input_ids.size(0), CFG.batch_size)
        bigbird_output = self.bigbird(input_ids, attention_mask, token_type_ids)[0]
        position_input = position_input.unsqueeze(-1)
        position_input = position_input.expand(batch, CFG.max_option_num, self.bigbird.config.vocab_size)
        out = torch.gather(bigbird_output, 1, position_input)
        out = out.view(batch, CFG.max_option_num, 1, self.bigbird.config.vocab_size)
        out = out.expand(batch, CFG.max_option_num, 4, self.bigbird.config.vocab_size)
        out = torch.gather(out, 3, options_input)
        out = out * options_mask
        out = out.sum(-1)
        out = out / (options_mask.sum(-1))
        out = out.view(-1, 4)

        if labels is not None:
            labels = labels.view(-1, )
            loss_fc = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fc(out, labels)
            loss = loss.view(batch, CFG.max_option_num)
            loss = loss * answers_mask
            loss = loss.sum() / (answers_mask.sum())

            acc = self.acc(out, labels)
            acc = acc.view(batch, CFG.max_option_num)
            acc = acc * answers_mask
            total_acc = acc.sum()

        # return loss, out, total_acc
        return loss, out, total_acc, acc

    def acc(self, out, tgt):
        out = torch.argmax(out, -1)
        return (out == tgt).float()


class BigBirdForSeqFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = CFG.num_labels
        self.bigbird = BigBirdForMaskedLM.from_pretrained(CFG.model_path)

        self.bert_head = nn.Sequential(
            nn.Linear(self.bigbird.config.hidden_size, self.bigbird.config.vocab_size),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, options_input, options_mask, answers_input,
                answers_mask, position_input, labels=None):
        loss = None
        total_acc = None
        batch = min(input_ids.size(0), CFG.batch_size)
        bigbird_output = self.bigbird(input_ids, attention_mask, token_type_ids)[0]
        position_input = position_input.unsqueeze(-1)
        position_input = position_input.expand(batch, CFG.max_option_num, self.bigbird.config.vocab_size)
        out = torch.gather(bigbird_output, 1, position_input)
        out = out.view(batch, CFG.max_option_num, 1, self.bigbird.config.vocab_size)
        out = out.expand(batch, CFG.max_option_num, 4, self.bigbird.config.vocab_size)
        out = torch.gather(out, 3, options_input)
        out = out * options_mask
        out = out.sum(-1)
        out = out / (options_mask.sum(-1))
        out = out.view(-1, 4)

        if labels is not None:
            labels = labels.view(-1, )
            loss_fc = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fc(out, labels)
            loss = loss.view(batch, CFG.max_option_num)
            loss = loss * answers_mask
            loss = loss.sum() / (answers_mask.sum())

            acc = self.acc(out, labels)
            acc = acc.view(batch, CFG.max_option_num)
            acc = acc * answers_mask
            total_acc = acc.sum()

        return loss, out, total_acc
        # return loss, out, total_acc, acc

    def acc(self, out, tgt):
        out = torch.argmax(out, -1)
        return (out == tgt).float()


class BigBirdForDistractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
        self.bigbird = BigBirdForMaskedLM.from_pretrained(CFG.model_path)
        self.bert_head = nn.Sequential(
            nn.Linear(self.bigbird.config.hidden_size, self.bigbird.config.vocab_size),
        )

    def forward(self, ):
        unmasker = pipeline('fill-mask', tokenizer=self.tokenizer, model=self.bigbird, top_k=100)
        return unmasker


class BigBirdForSeq4option(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = CFG.num_labels
        self.bigbird = BigBirdModel.from_pretrained(CFG.model_path)
        self.dropout = nn.Dropout(CFG.dropout)

        self.bert_head = nn.Sequential(
            nn.Linear(self.bigbird.config.hidden_size, self.bigbird.config.vocab_size),
        )

        self.linear_head = nn.Sequential(
            nn.Dropout(CFG.dropout),
            nn.Linear(self.bigbird.config.hidden_size, 1)
        )

        self.vocab_embedding = nn.Embedding(CFG.len_vocab, self.bigbird.config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, options_input, position_input, answers_mask,
                opt_len, labels=None):
        loss = None
        total_acc = None
        batch = min(input_ids.size(0), CFG.batch_size)
        bigbird_output = self.bigbird(input_ids, attention_mask, token_type_ids)[0]
        position_input = position_input.unsqueeze(-1)
        position_input = position_input.expand(batch, CFG.max_option_num, self.bigbird.config.hidden_size)
        out = torch.gather(bigbird_output, 1, position_input)
        out = out.reshape(-1, 768)
        options_input = options_input.reshape(-1, 4)
        options_embeddings = self.vocab_embedding(options_input)
        # options_embeddings = self.vocab_embedding(options_input).transpose(1, 2)
        # logits = torch.einsum('ab,abc->ac', out, options_embeddings)

        logits = torch.einsum('abc,ac->abc', [options_embeddings, out])
        logits = self.linear_head(logits)
        logits = logits.view(-1, 4)
        logits = logits.view(batch, CFG.max_option_num, 4)

        if labels is not None:
            labels = labels.view(-1,)
            loss_fc = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fc(logits, labels)
            loss = loss.view(batch, CFG.max_option_num)
            loss = loss * answers_mask
            loss = loss.sum() / (answers_mask.sum())

            acc = self.acc(logits, labels)
            acc = acc.view(batch, CFG.max_option_num)
            acc = acc * answers_mask
            total_acc = acc.sum()
        return loss, logits, total_acc

    def acc(self, out, tgt):
        out = torch.argmax(out, -1)
        return (out == tgt).float()


class BigBirdForSeq_4opt(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = CFG.num_labels
        self.bigbird = BigBirdForMaskedLM.from_pretrained(CFG.model_path)

        self.bert_head = nn.Sequential(
            nn.Linear(self.bigbird.config.hidden_size, self.bigbird.config.vocab_size),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, options_input, options_mask, answers_input,
                answers_mask, position_input, labels=None, level=None, train=True):
        loss = None
        total_acc = None
        batch = min(input_ids.size(0), CFG.batch_size)
        bigbird_output = self.bigbird(input_ids, attention_mask, token_type_ids)[0]
        position_input = position_input.unsqueeze(-1)
        position_input = position_input.expand(batch, CFG.max_option_num, self.bigbird.config.vocab_size)
        out = torch.gather(bigbird_output, 1, position_input)
        out = out.view(batch, CFG.max_option_num, 1, self.bigbird.config.vocab_size)
        out = out.expand(batch, CFG.max_option_num, 4, self.bigbird.config.vocab_size)
        out = torch.gather(out, 3, options_input)
        out = out * options_mask
        out = out.sum(-1)
        out = out / (options_mask.sum(-1))
        out = out.view(-1, 4)

        if labels is not None:
            labels = labels.view(-1, )
            loss_fc = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fc(out, labels)
            loss = loss.view(batch, CFG.max_option_num)
            loss = loss * answers_mask
            loss = loss.sum() / (answers_mask.sum())

            acc = self.acc(out, labels)
            acc = acc.view(batch, CFG.max_option_num)
            acc = acc * answers_mask
            total_acc = acc.sum()

        return loss, out, total_acc

    def acc(self, out, tgt):
        out = torch.argmax(out, -1)
        return (out == tgt).float()






