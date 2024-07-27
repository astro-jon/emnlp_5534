import pandas as pd


file_path = '../data_emnlp_/emnlp_bigbird_option_pos_middle_orig_confidence_easy/'
result_df = pd.DataFrame(columns=['item_id', 'sentence_mask', 'options', 'opt_len', 'opt_pos', 'answer', 'answer_word'])
for i in range(5):
    data = pd.read_csv(file_path + f'fold_{i}/test.csv')
    result_df = pd.concat([result_df, data], axis=0, ignore_index=True)
    print()
result_df.to_csv('../data_emnlp_/middle_easy_confidence.csv', index=False)



























