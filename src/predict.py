import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer, AutoTokenizer
from cfg import CFG
from transformers.trainer_pt_utils import *
from transformers.integrations import *
import time
import torch
import numpy as np
from dataset import InputDataset, InputDatasetOption, InputDatasetWithVocab, InputDatasetPromptAfter
from model import BigBirdForSeq
from Untitled import *
from transformers.utils.notebook import format_time
from torch.cuda.amp import autocast

os.environ["WANDB_DISABLED"] = "true"


def evaluate(model, data_loader):
    total_val_loss = 0
    corrects = []
    predict_list = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(CFG.device)
        attention_mask = batch['attention_mask'].to(CFG.device)
        token_type_ids = batch['token_type_ids'].to(CFG.device)
        options_input = batch['options_input'].to(CFG.device)
        options_mask = batch['options_mask'].to(CFG.device)
        answers_input = batch['answers_input'].to(CFG.device)
        answers_mask = batch['answers_mask'].to(CFG.device)
        position_input = batch['position_input'].to(CFG.device)
        labels = batch['label'].to(CFG.device)
        with torch.no_grad():
            loss, output, acc = model(input_ids, attention_mask, token_type_ids, options_input,
                                      options_mask, answers_input, answers_mask, position_input, labels)
        total_val_loss += loss.item()
        gap_num = answers_mask.sum()
        corrects.append((acc / gap_num).item())
    avg_val_loss = total_val_loss / len(data_loader)
    avg_val_acc = np.mean(corrects)

    return avg_val_loss, avg_val_acc


def main():
    seed_everything(CFG.seed)
    log = log_creater(output_dir='../cache/logs_m/')
    log.info(CFG.name)
    log.info('EPOCH = {}; LR = {}'.format(CFG.epochs, CFG.learning_rate))
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
    result = []
    for fold in range(5):
        test_data = pd.read_csv('../data/bigbird_sentence_middle/fold_{}/test.csv'.format(fold))
        te_dataset = InputDatasetOption(test_data, tokenizer, CFG.max_input_length)
        te_data_loader = DataLoader(te_dataset, batch_size=CFG.batch_size, shuffle=False)
        model = BigBirdForSeq()
        model.to(CFG.device)
        model.eval()

        _, test_acc = evaluate(model, te_data_loader)
        log.info('avg_test_acc={:.5f}===='.format(test_acc))
        log.info('==========================')
        result.append(test_acc)
    str_result = str(result)
    log.info('5 fold result: {}'.format(str_result))
    log.info('mean test acc: {:.5f}'.format(np.mean(result)))


if __name__ == '__main__':
    main()














