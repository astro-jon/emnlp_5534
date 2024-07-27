import pandas as pd
import numpy as np
from tqdm import tqdm

max_sentence_len = 0
max_opt_len = 0
max_opt_num = 0
sent = None
opt = None
splits = ['train', 'val', 'test']

for fold in range(1):
    for split in splits:
        data = pd.read_csv('../data/bigbird_sentence_high/fold_{}/{}.csv'.format(fold, split))
        for _, row in tqdm(data.iterrows(), total=len(data)):
            text = eval(row['sentence_mask'])
            if len(text) > max_sentence_len:
                max_sentence_len = len(text)
                opt = row['options']
                sent = text
            opt_num = eval(row['options'])
            if len(opt_num) > max_opt_num:
                max_opt_num = len(opt_num)
            opt_len = eval(row['opt_len'])
            if np.max(opt_len) > max_opt_len:
                max_opt_len = np.max(opt_len)
                # opt = row['options']

print(max_sentence_len, max_opt_num, max_opt_len)
print(sent)
print(opt)




































