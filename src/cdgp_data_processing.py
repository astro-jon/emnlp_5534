import sys
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer, AutoTokenizer, BigBirdForMaskedLM, pipeline
from cfg import CFG
from transformers.trainer_pt_utils import *
from transformers.integrations import *
import time
import torch
import numpy as np
from dataset import InputDatasetOptionFeature
from model import BigBirdForSeqFeature, BigBirdForDistractor
from Untitled import *
from transformers.utils.notebook import format_time
from torch.cuda.amp import autocast
import fasttext
import Levenshtein
import nltk
from nltk.tokenize import word_tokenize
import random

# def evaluate(model, data_loader):
#     result_df = pd.DataFrame(
#         columns=['item_id', 'softmax', 'shannonent', 'avg_word_embedding', 'avg_sentence_embedding',
#                  'lenvenshtein', 'pos'])
#     for batch in tqdm(data_loader):
#         item_id = batch['item_id']
#         # for i in item_id:
#         #     item_id_list.append(i)
#         input_ids = batch['input_ids'].to(CFG.device)
#         attention_mask = batch['attention_mask'].to(CFG.device)
#         token_type_ids = batch['token_type_ids'].to(CFG.device)
#         options_input = batch['options_input'].to(CFG.device)
#         options_mask = batch['options_mask'].to(CFG.device)
#         answers_input = batch['answers_input'].to(CFG.device)
#         answers_mask = batch['answers_mask'].to(CFG.device)
#         position_input = batch['position_input'].to(CFG.device)
#         labels = batch['label'].to(CFG.device)
#         orl_sentence = batch['orl_sentence']
#         orl_option = [eval(i) for i in batch['orl_option']]
#         answer_word = batch['answer_word']
#         with torch.no_grad():
#             loss, output, total_acc = model(input_ids, attention_mask, token_type_ids, options_input, options_mask,
#                                             answers_input, answers_mask, position_input, labels)
#         output = torch.softmax(output, dim=1).detach().cpu().tolist()
#         labels = labels.detach().cpu().tolist()
#         word_similarities = list()
#         sent_similarities = list()
#         for index in range(len(item_id)):
#             # Word Embedding Similarity
#             answer_vector = ds_model.get_word_vector(answer_word[index])
#             for c in orl_option[index]:
#                 c_vector = ds_model.get_word_vector(c)
#                 word_similarity = similarity(answer_vector, c_vector)  # Cosine similarity between A and Di
#                 word_similarities.append(word_similarity)
#
#             # Contextual-Sentence Embedding Similarity s2
#             correct_sent = orl_sentence[index].replace('[MASK]', answer_word[index])
#             correct_sent_vector = ds_model.get_sentence_vector(correct_sent)
#             cand_sents = list()
#             for c in orl_option[index]:
#                 cand_sents.append(orl_sentence[index].replace('[MASK]', c))
#             for cand_sent in cand_sents:
#                 cand_sent_vector = ds_model.get_sentence_vector(cand_sent)
#                 sent_similarity = similarity(correct_sent_vector,
#                                              cand_sent_vector)  # Cosine similarity between S(A) and S(Di)
#                 sent_similarities.append(sent_similarity)
#
#         whole_options = []
#         for i in range(len(orl_option)):
#             orl_option[i].insert(labels[i][0], answer_word[i])
#         print()
#
#
# # Cosine similarity
# def similarity(v1, v2):
#     n1 = np.linalg.norm(v1)
#     n2 = np.linalg.norm(v2)
#     if n1 == 0 or n2 == 0:  # Denominator can not be zero
#         return 1
#     else:
#         return np.dot(v1, v2) / (n1 * n2)
#
#
# def main():
#     tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
#     all_result_df = pd.DataFrame()
#     for fold in range(5):
#         test_data = pd.read_csv('../data/bigbird_middle/fold_{}/test.csv'.format(fold))
#         te_dataset = InputDatasetOptionFeature(test_data, tokenizer, CFG.max_input_length)
#         te_data_loader = DataLoader(te_dataset, batch_size=CFG.batch_size, shuffle=False)
#         best_model_path = CFG.save_path + f'{CFG.name}_batch{CFG.batch_size}_epoch{CFG.epochs}_lr{CFG.learning_rate}/'
#         model = BigBirdForSeqFeature()
#         model.to(CFG.device)
#         model.eval()
#         model.load_state_dict(torch.load(best_model_path + 'fold{}.pt'.format(fold)))
#         result_df = evaluate(model, te_data_loader)
#         all_result_df = pd.concat([all_result_df, result_df])
#
#
# if __name__ == '__main__':
#     DS_MODEL = "/media/disk3/X/假期/整理/cloze_question_answer/final_bigbird/cdgp-ds-fasttext.bin"
#     ds_model = fasttext.load_model(DS_MODEL)
#     main()


def evaluate(model, data_loader):
    all_distractors = []
    all_dist_len = []
    electra_all_distractors = []
    electra_all_dist_len = []
    for batch in tqdm(data_loader):
        labels = batch['label'].to(CFG.device)
        orl_sentences = batch['orl_sentence']
        answer_word = batch['answer_word']
        for index in range(len(orl_sentences)):
            unmasker = model()
            distractors = []
            dist_len = []
            electra_distractors = []
            electra_dist_len = []
            for i in unmasker(orl_sentences[index]):
                distractor = i["token_str"].replace(" ", "")
                electra_distractor = i["token_str"].replace(" ", "")
                if distractor == '<unk>':
                    continue
                if distractor != answer_word[index] and len(distractors) < 3:
                    distractor = tokenizer.tokenize(distractor)
                    electra_distractor = electra_tokenizer.tokenize(electra_distractor)
                    if distractor not in distractors:
                        distractors.append(distractor)
                        dist_len.append(len(distractor))
                    if electra_distractor not in electra_distractors:
                        electra_distractors.append(electra_distractor)
                        electra_dist_len.append(len(electra_distractor))
                elif len(distractors) == 3:
                    break
            random.shuffle(distractors)
            dist_len = [x for _, x in sorted(zip(distractors, dist_len))]
            token_answer = tokenizer.tokenize(answer_word[index])
            distractors.insert(labels[index][0], token_answer)
            all_distractors.append(distractors)
            dist_len.insert(labels[index][0], len(token_answer))
            all_dist_len.append(dist_len)

            random.shuffle(electra_distractors)
            electra_dist_len = [x for _, x in sorted(zip(electra_distractors, electra_dist_len))]
            electra_token_answer = electra_tokenizer.tokenize(answer_word[index])
            electra_distractors.insert(labels[index][0], electra_token_answer)
            electra_all_distractors.append(electra_distractors)
            electra_dist_len.insert(labels[index][0], len(electra_token_answer))
            electra_all_dist_len.append(electra_dist_len)
    return all_distractors, all_dist_len, electra_all_distractors, electra_all_dist_len


def main():
    for fold in range(5):
        test_data = pd.read_csv('../data/bigbird_middle/fold_{}/test.csv'.format(fold))
        electra_test_data = pd.read_csv('../data/electra_middle/fold_{}/test.csv'.format(fold))
        te_dataset = InputDatasetOptionFeature(test_data, tokenizer, CFG.max_input_length)
        te_data_loader = DataLoader(te_dataset, batch_size=CFG.batch_size, shuffle=False)
        best_model_path = CFG.save_path + f'{CFG.name}_batch{CFG.batch_size}_epoch{CFG.epochs}_lr{CFG.learning_rate}/'
        model = BigBirdForDistractor()
        # model.to(CFG.device)
        model.eval()
        model.load_state_dict(torch.load(best_model_path + 'fold{}.pt'.format(fold)))
        all_distractors, all_dist_len, electra_all_distractors, electra_all_dist_len = evaluate(model, te_data_loader)
        test_data['options'] = all_distractors
        test_data['opt_len'] = all_dist_len
        electra_test_data['options'] = electra_all_distractors
        electra_test_data['opt_len'] = electra_all_dist_len
        save_path = f'../data/new_option_middle_1e-4_16/fold_{fold}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        electra_save_path = f'../data/electra_new_option_middle_1e-4_16/fold_{fold}/'
        if not os.path.exists(electra_save_path):
            os.makedirs(electra_save_path)
        test_data.to_csv(save_path+'test.csv', sep=',', index=False)
        electra_test_data.to_csv(electra_save_path + 'test.csv', sep=',', index=False)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
    electra_tokenizer = AutoTokenizer.from_pretrained('../model/electra-base-generator')
    main()
















