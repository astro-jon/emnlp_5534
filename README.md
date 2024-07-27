## Code Project

- [bert_base_uncased](bert_base_uncased)
  - dir save the model

- [data](data): data dir
  - *_orig:  
    - fold_*: 5 fold of data
      - train: used for finetuning bert to assess certain profess 
      - valid: for valid the model's ability to save the best checkpoint model
      - test: original test question
      
  - *_easy: `easy-level` cloze-item 
    - fold_*: 5 fold of data
      - test: generated easy level question
  - *_hard: `hard-level` cloze-item
    - fold_*: 5 fold of data
      - test: generated hard level question 


| item_id | sentence_mask | options | opt_len | opt_pos |       answer        | answer_word |
| :---: | :---: | :---: | :---: | :---: |:-------------------:| :---: |
| the id of each item | masked sentence | each options | the tokenized lenn of each option | option position in item | answer('a','b',...) | answer word|


- [src](src): script of project
  - [predict_to_json.py](src%2Fpredict_to_json.py): run this code to answer question and save the response for IRT
