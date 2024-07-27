import logging
import os
import random
import time
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)

    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))

    return log


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_csv_data(data_path):
    min_max = MinMaxScaler(feature_range=(0, 1))
    data_df = pd.read_csv(data_path)
    data_vec = data_df.values
    data_vec = np.hstack((data_vec[:, 0].reshape(-1, 1), min_max.fit_transform(data_vec[:, 1:])))
    return data_vec


def feature_to_tensor(feature, text_id):
    feature_list = []
    for i in text_id:
        for j in feature:
            if i.item() == int(j[0]):
                feature_list.append(j[1:].tolist())
                break
        else:
            print(i)
    feature_torch = torch.Tensor(feature_list)
    return feature_torch


def gaussian_feature_to_tensor(feature, text):
    feature = feature.to_numpy()
    feature_list = []
    for i in text:
        for j in feature:
            if i == j[1]:
                feature_list.append(j[2:feature.shape[1]].tolist())
                break
        else:
            print(i)
    feature_torch = torch.Tensor(feature_list)
    return feature_torch


def read_data(data_path):
    data_df = pd.read_csv(data_path, header=None)
    data_vec = data_df.values[:, 1:]
    min_max = MinMaxScaler(feature_range=(0, 1))
    data_vec = min_max.fit_transform(data_vec)
    item_id_df = data_df[[0]]
    item_id_df.rename(columns={0: 'item_id'}, inplace=True)
    data_vec = pd.concat([item_id_df, pd.DataFrame(data_vec)], axis=1)
    return data_vec


def find_opt(item_id):
    data_df = pd.read_csv('./data/article_data.csv')
    for index, row in data_df.iterrows():
        if item_id == row['item_id']:
            distractors = ' '.join(' '.join(i) for i in eval(row['distractors']))
            return distractors


def check_text_option(df, tokenizer):
    result_df = pd.DataFrame(columns=('item_id', 'text', 'distractor', 'level'))
    data_df = pd.read_csv('./data/article_data.csv')
    for index, row in tqdm(df.iterrows(), total=len(df)):
        item_id = row['item_id']
        text = row['text']
        level = row['level']
        for _, row1 in data_df.iterrows():
            if item_id == row1['item_id']:
                distractors = ' '.join(' '.join(i) for i in eval(row1['distractors']))
                break
        text_token = tokenizer.tokenize(text)
        dis_token = tokenizer.tokenize(distractors)
        all_len = len(text_token) + len(dis_token) + 3
        if all_len > 512:
            first_text = text_token[:len(text_token)-(all_len-512)]
            second_text = text_token[-(len(text_token)-(all_len-512)):]
            result_df.loc[len(result_df)] = [item_id, first_text, dis_token, level]
            result_df.loc[len(result_df)] = [item_id, second_text, dis_token, level]
        else:
            result_df.loc[len(result_df)] = [item_id, text_token, dis_token, level]
    return result_df









