import torch


class CFG:
    save_path = '../best_model/'  # 模型训练结果
    name = 'bigbird_cloze_sep_opt'
    model_path = '/home/a401/桌面/办公/xie/tidy/bigbird_qa/model/bigbird-roberta-base'
    num_labels = 3
    num_feature = 43
    max_input_length = 256
    epochs = 1
    learning_rate = 3e-5   # 1e-3
    dropout = 0.3
    hidden_size = 256
    warmup_proportion = 0.1
    batch_size = 16
    seed = 2022
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_option_num = 1
    max_option_len = 20
    len_vocab = 15171



