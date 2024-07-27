import glob
import json
import os

import jsonlines
import numpy as np

if __name__ == '__main__':
    fold = 'middle_irt_format'
    os.makedirs(fold, exist_ok = True)
    for file in glob.glob('../data/model_answer/*'):
        reader = open(file, 'r', encoding = 'utf-8').readlines()
        _, fileName = os.path.split(file)
        subject_id_list = []
        response_list = []
        all_true = 0
        for line in reader:
            line = json.loads(line)
            subject = [i for i in line.keys()][0]
            subject_id_list.append(subject)
            qid2response = [i for i in line.values()][0]
            question_list = [i for i in qid2response.keys()]
            response_list.append([i for i in qid2response.values()])
            # new_dict = {
            #     "subject_id": subject, "responses": response
            # }
        writer = jsonlines.open(f'{fold}/{fileName}', 'w')
        processed_response = []
        valid_qid_list = []
        response_list = np.array(response_list)
        for row, qid in zip(response_list.T, question_list):  # 去除全对或全错的数据
            if np.sum(row) == len(row):
                all_true += 1
            if (np.sum(row) == 0) or (np.sum(row) == len(row)):
                continue
            processed_response.append(row)
            valid_qid_list.append(qid)
        print(f'有效比例为：{len(valid_qid_list) / len(question_list)}; 全对为{ len(valid_qid_list) / all_true}')
        processed_response = np.array(processed_response).T
        for idx, subject in enumerate(subject_id_list):
            response_dict = {}
            for opt, res in zip(valid_qid_list, processed_response[idx]):
                response_dict.update({opt: int(res)})
            writer.write({
                "subject_id": subject,
                "responses": response_dict
            })
        writer.close()