import json
import os.path

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
            loss, output, total_acc, acc_list = model(input_ids, attention_mask, token_type_ids, options_input,
                                                      options_mask, answers_input, answers_mask, position_input, labels)
        total_val_loss += loss.item()
        gap_num = answers_mask.sum()
        acc_list = acc_list[:int(gap_num)]
        corrects.append((total_acc / gap_num).item())
        try:
            predict_list += acc_list.squeeze().tolist()
        except:
            predict_list.append(acc_list.squeeze().tolist())
    avg_val_loss = total_val_loss / len(data_loader)
    avg_val_acc = np.mean(corrects)

    return avg_val_loss, avg_val_acc, predict_list


def main():
    seed_everything(CFG.seed)
    log = log_creater(output_dir='../cache/logs_m/')
    log.info(CFG.name)
    log.info('EPOCH = {}; LR = {}'.format(CFG.epochs, CFG.learning_rate))
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)

    result = []
    para_dict = [(1e-4, 16), (1e-4, 32), (1e-5, 16), (1e-5, 32), (3e-5, 16), (3e-5, 32)]
    para_dictMiddle = [(1e-4, 32)]
    for para in para_dictMiddle:
        for fold in range(1):
            test_data = pd.read_csv('../data_3_different/bigbird_clozeRanking_optionToken/fold1_geneNew_bigbird.csv')
            te_dataset = InputDatasetOption(test_data, tokenizer, CFG.max_input_length)
            te_data_loader = DataLoader(te_dataset, batch_size=CFG.batch_size, shuffle=False)
            #
            best_model_path = CFG.save_path + f'{CFG.name}_batch{para[1]}_epoch{CFG.epochs}_lr{para[0]}/'
            model = BigBirdForSeq()
            model.to(CFG.device)
            model.eval()
            model.load_state_dict(torch.load(best_model_path + 'fold{}.pt'.format(fold)))
            json_model_name = CFG.save_path + f'{CFG.name}_batch{para[1]}_epoch{CFG.epochs}_lr{para[0]}/'
            _, test_acc, acc_list = evaluate(model, te_data_loader)
            acc_dict = {}
            for qid, cor in enumerate(acc_list):
                acc_dict[str(qid)] = cor
            save_path = '../data_3_different/model_answer_middle_clozeRanking_optionToken_fold1/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # os.makedirs(save_path)
            writer = open(save_path + f'fold{fold}.jsonlines', 'a+', encoding='utf8')
            writer.write(
                json.dumps({
                    json_model_name: acc_dict
                }) + '\n'
            )
            result.append(test_acc)
        str_result = str(result)
        log.info('5 fold result: {}'.format(str_result))
        log.info('mean test acc: {:.5f}'.format(np.mean(result)))


if __name__ == '__main__':
    main()























































