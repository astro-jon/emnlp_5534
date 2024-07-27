import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from transformers import get_linear_schedule_with_warmup, BertTokenizer, AutoTokenizer
from cfg import CFG
from transformers.trainer_pt_utils import *
from transformers.integrations import *
import time
import torch
import numpy as np
from dataset import InputDataset, InputDatasetOption, InputDatasetPromptAfter, InputDatasetPromptBefore
from model import BigBirdForSeq
from Untitled import *
from transformers.utils.notebook import format_time
from torch.cuda.amp import autocast


os.environ["WANDB_DISABLED"] = "true"


def train(model, fold, train_loader, val_loader, test_loader, log):
    weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
    bias_params = [param for name, param in model.named_parameters() if "bias" in name]
    optimizer = AdamW([{'params': weight_params, 'weight_decay': 1e-5},
                       {'params': bias_params, 'weight_decay': 0}],
                      lr=CFG.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    total_steps = len(train_loader) * CFG.epochs
    best_model_path = CFG.save_path + f'{CFG.name}_batch{CFG.batch_size}_epoch{CFG.epochs}_lr{CFG.learning_rate}/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    log.info("Train batch size = {}".format(CFG.batch_size))
    log.info("Total steps = {}".format(total_steps))
    log.info("Training Start!")
    log.info('')

    best_val_loss = float('inf')

    for epoch in range(CFG.epochs):
        total_train_loss = 0
        t0 = time.time()
        model.to(CFG.device)
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            token_type_ids = batch['token_type_ids'].to(CFG.device)
            options_input = batch['options_input'].to(CFG.device)
            options_mask = batch['options_mask'].to(CFG.device)
            answers_input = batch['answers_input'].to(CFG.device)
            answers_mask = batch['answers_mask'].to(CFG.device)
            position_input = batch['position_input'].to(CFG.device)
            labels = batch['label'].to(CFG.device)
            model.zero_grad()
            with autocast():
                loss, output, acc = model(input_ids, attention_mask, token_type_ids, options_input, options_mask,
                                          answers_input, answers_mask, position_input, labels)
            total_train_loss += loss.item()

            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            # optimizer.step()
            gap_num = answers_mask.sum()
            if step % 20 == 0:
                print('step : {},   loss : {},   acc : {}'.format(step, loss.item(), (acc/gap_num).item()))
        avg_train_loss = total_train_loss / len(train_loader)
        train_time = format_time(time.time() - t0)

        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch + 1, CFG.epochs, avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')

        model.eval()
        avg_val_loss, avg_val_acc = evaluate(model, val_loader)
        val_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f}===='.format(epoch + 1, CFG.epochs,
                                                                                       avg_val_loss, avg_val_acc))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path + 'fold{}.pt'.format(fold))
            print('Model Saved!')
    log.info('   Training Completed!')
    log.info('')
    model.load_state_dict(torch.load(best_model_path + 'fold{}.pt'.format(fold)))
    _, test_acc = evaluate(model, test_loader)
    log.info('avg_test_acc={:.5f}===='.format(test_acc))
    log.info('==========================')
    return test_acc


def evaluate(model, data_loader):
    total_val_loss = 0
    corrects = []
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
        # level = batch['level'].to(CFG.device)

        with torch.no_grad():
            loss, output, acc = model(input_ids, attention_mask, token_type_ids, options_input, options_mask,
                                      answers_input, answers_mask, position_input, labels)

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
        tr_data = pd.read_csv('../data/bigbird_sentence_middle/fold_{}/train.csv'.format(fold))
        va_data = pd.read_csv('../data/bigbird_sentence_middle/fold_{}/val.csv'.format(fold))
        test_data = pd.read_csv('../data/bigbird_sentence_middle/fold_{}/test.csv'.format(fold))

        tr_dataset = InputDatasetOption(tr_data, tokenizer, CFG.max_input_length)
        tr_data_loader = DataLoader(tr_dataset, batch_size=CFG.batch_size, shuffle=True)
        va_dataset = InputDatasetOption(va_data, tokenizer, CFG.max_input_length)
        va_data_loader = DataLoader(va_dataset, batch_size=CFG.batch_size, shuffle=False)
        te_dataset = InputDatasetOption(test_data, tokenizer, CFG.max_input_length)
        te_data_loader = DataLoader(te_dataset, batch_size=CFG.batch_size, shuffle=False)

        model = BigBirdForSeq()

        test_acc = train(model, fold, tr_data_loader, va_data_loader, te_data_loader, log)

        result.append(test_acc)
    str_result = str(result)
    log.info('5 fold result: {}'.format(str_result))
    log.info('mean test acc: {:.5f}'.format(np.mean(result)))


if __name__ == '__main__':
    main()


