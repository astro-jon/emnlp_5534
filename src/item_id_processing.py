import json
import os
from tqdm import tqdm
import pandas as pd

data_path = '../data/bigbird_sentence_middle/'
splits = ['train', 'val', 'test']
for i in range(5):
    for split in splits:
        df = pd.read_csv(data_path + f'fold_{i}/{split}.csv')
        with open(f'../data/cloth_sentence/middle/fold_{i}/{split}.json', 'r', encoding='utf8') as f:
            json_data = json.load(f)

        count = 0
        temp = None
        new_item_id = None
        item_id_list = []
        options_list = []
        answers_word_list = []
        sentence_list = []

        for index, row in tqdm(df.iterrows(), total=len(df)):
            item_id = row['item_id']
            if temp == item_id or count == 0:
                new_item_id = str(item_id) + '_' + str(count)
                count += 1
                temp = item_id
            elif temp != item_id:
                count = 0
                new_item_id = str(item_id) + '_' + str(count)
                count += 1
                temp = item_id
            item_id_list.append(new_item_id)
            options_list.append(json_data[index]['distractors'])
            answers_word_list.append(json_data[index]['answer'])
            sentence_list.append(json_data[index]['sentence'])

        df['item_id'] = item_id_list
        df['orl_options'] = options_list
        df['answer_word'] = answers_word_list
        df['orl_sentence'] = sentence_list
        save_path = f'../data/middle/fold_{i}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(save_path+f'{split}.csv', sep=',', index=False)
        print()




















