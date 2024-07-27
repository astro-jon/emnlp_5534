import sys
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
from dataset import InputDatasetOptionFeature
from model import BigBirdForSeqFeature
from Untitled import *
from transformers.utils.notebook import format_time
from torch.cuda.amp import autocast
import fasttext
import Levenshtein
import nltk
from nltk.tokenize import word_tokenize


os.environ["WANDB_DISABLED"] = "true"


def evaluate(model, data_loader):
    total_val_loss = 0
    corrects = []
    predict_list = []
    result_df = pd.DataFrame(columns=['item_id', 'softmax', 'shannonent', 'avg_word_embedding', 'avg_sentence_embedding',
                                      'lenvenshtein', 'pos'])
    item_id_list = []
    softmax_list = []
    shannonent_list = []
    avg_word_embedding_list = []
    avg_sentence_embedding_list = []
    lenvenshtein_list = []
    pos_list = []

    for batch in tqdm(data_loader):
        item_id = batch['item_id']
        for i in item_id:
            item_id_list.append(i)
        input_ids = batch['input_ids'].to(CFG.device)
        attention_mask = batch['attention_mask'].to(CFG.device)
        token_type_ids = batch['token_type_ids'].to(CFG.device)
        options_input = batch['options_input'].to(CFG.device)
        options_mask = batch['options_mask'].to(CFG.device)
        answers_input = batch['answers_input'].to(CFG.device)
        answers_mask = batch['answers_mask'].to(CFG.device)
        position_input = batch['position_input'].to(CFG.device)
        labels = batch['label'].to(CFG.device)
        orl_sentence = batch['orl_sentence']
        answer_word = batch['answer_word']
        with torch.no_grad():
            loss, output, total_acc = model(input_ids, attention_mask, token_type_ids, options_input, options_mask,
                                            answers_input, answers_mask, position_input, labels)
        output = torch.softmax(output, dim=1).detach().cpu().tolist()
        orl_option = [eval(i) for i in batch['orl_option']]
        shannonent = [calcShannonEnt(i) for i in output]
        for i in output:
            softmax_list.append(i)
        for i in shannonent:
            shannonent_list.append(i)

        for index in range(len(item_id)):
            # Word Embedding Similarity
            answer_vector = ds_model.get_word_vector(answer_word[index])
            word_similarities = list()
            for c in orl_option[index]:
                c_vector = ds_model.get_word_vector(c)
                word_similarity = similarity(answer_vector, c_vector)  # Cosine similarity between A and Di
                word_similarities.append(word_similarity)
            avg_word_embedding_list.append(np.mean([1-i for i in word_similarities]))

            # Contextual-Sentence Embedding Similarity s2
            correct_sent = orl_sentence[index].replace('[MASK]', answer_word[index])
            correct_sent_vector = ds_model.get_sentence_vector(correct_sent)
            cand_sents = list()
            for c in orl_option[index]:
                cand_sents.append(orl_sentence[index].replace('[MASK]', c))
            sent_similarities = list()
            for cand_sent in cand_sents:
                cand_sent_vector = ds_model.get_sentence_vector(cand_sent)
                sent_similarity = similarity(correct_sent_vector,
                                             cand_sent_vector)  # Cosine similarity between S(A) and S(Di)
                sent_similarities.append(sent_similarity)
            avg_sentence_embedding_list.append(np.mean([1-i for i in sent_similarities]))

            levenshtein_distance = list()
            for c in orl_option[index]:
                leven_sim = Levenshtein.jaro(answer_word[index], c)
                levenshtein_distance.append(leven_sim)
            lenvenshtein_list.append(np.mean([i for i in levenshtein_distance]))

            # POS match score s3
            origin_token = word_tokenize(orl_sentence[index])
            origin_token.remove("[")
            origin_token.remove("]")

            mask_index = origin_token.index("MASK")

            correct_token = word_tokenize(correct_sent)
            correct_pos = nltk.pos_tag(correct_token)
            answer_pos = correct_pos[mask_index]  # POS of A

            pos_count = 0
            for i in cand_sents:
                cand_sent_token = word_tokenize(i)
                cand_sent_pos = nltk.pos_tag(cand_sent_token)
                cand_pos = cand_sent_pos[mask_index]  # POS of Di
                if cand_pos[1] == answer_pos[1]:
                    pos_count += 1
            pos_list.append(pos_count/3)
    result_df['item_id'] = item_id_list
    result_df['softmax'] = softmax_list
    result_df['shannonent'] = shannonent_list
    result_df['avg_word_embedding'] = avg_word_embedding_list
    result_df['avg_sentence_embedding'] = avg_sentence_embedding_list
    result_df['lenvenshtein'] = lenvenshtein_list
    result_df['pos'] = pos_list
    return result_df


def calcShannonEnt(data):
    shannonEnt = 0.0
    for prob in data:
        if prob == 0:
            continue
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


# Cosine similarity
def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:  # Denominator can not be zero
        return 1
    else:
        return np.dot(v1, v2) / (n1 * n2)


# Min–max normalization
def min_max_y(raw_data):
    min_max_data = []
    # Min–max normalization
    if max(raw_data) - min(raw_data) == 0:
        return raw_data
    for d in raw_data:
        min_max_data.append((d - min(raw_data)) / (max(raw_data) - min(raw_data)))

    return min_max_data


def main():
    seed_everything(CFG.seed)
    log = log_creater(output_dir='../cache/logs_m/')
    log.info(CFG.name)
    log.info('EPOCH = {}; LR = {}'.format(CFG.epochs, CFG.learning_rate))
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
    result = []
    all_result_df = pd.DataFrame()
    for fold in range(5):
        test_data = pd.read_csv('../data/bigbird_middle/fold_{}/test.csv'.format(fold))
        te_dataset = InputDatasetOptionFeature(test_data, tokenizer, CFG.max_input_length)
        te_data_loader = DataLoader(te_dataset, batch_size=CFG.batch_size, shuffle=False)
        best_model_path = CFG.save_path + f'{CFG.name}_batch{CFG.batch_size}_epoch{CFG.epochs}_lr{CFG.learning_rate}/'
        model = BigBirdForSeqFeature()
        model.to(CFG.device)
        model.eval()
        model.load_state_dict(torch.load(best_model_path + 'fold{}.pt'.format(fold)))
        result_df = evaluate(model, te_data_loader)
        all_result_df = pd.concat([all_result_df, result_df])
        print()
    print()
    all_result_df.to_csv(f'../data/bigbird_bs{CFG.batch_size}_lr{CFG.learning_rate}.csv', sep=',', index=False)


if __name__ == '__main__':
    DS_MODEL = "/media/disk3/X/假期/整理/cloze_question_answer/final_bigbird/cdgp-ds-fasttext.bin"
    ds_model = fasttext.load_model(DS_MODEL)
    main()


















