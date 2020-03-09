import os
import json

import torch

from EmotionX_KU_master.hparams import EMOTIONX_MODEL_HPARAMS
from pytorch_pretrained_bert import BertTokenizer
from EmotionX_KU_master.models import EmotionX_Model
from EmotionX_KU_master.utils import get_batch


#
# def set_n_appear(n_appear):
#     hparams.n_appear=n_appear

def label(test_json_path, model, n_appear, pretrained_model_path=None, name=None, email=None):
    if not torch.cuda.is_available():
        raise NotImplementedError()
    hparams = type('', (object,), EMOTIONX_MODEL_HPARAMS)()  # dict to class
    # hparams.n_appear = [65508, 15240, 4048, 3596, 28552]  # not to be used
    # hparams.n_appear = [46, 9, 6, 3, 32]  # not to be used

    print('preprocessing...')
    tokenizer = BertTokenizer.from_pretrained(hparams.bert_type)
    json_data = json.loads(open(test_json_path, 'r', encoding='utf-8').read())
    inputs = []
    for dialog in json_data:
        tokenized_dialog = []
        for utter_dict in dialog:
            tokenized_utter = tokenizer.tokenize(utter_dict['utterance'].lower())
            tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized_utter)
            if len(tokenized_dialog + tokenized_utter) + 1 > hparams.max_input_len:
                print('[CAUTION] over max_input_len: ', utter_dict['utterance'])
                continue
            tokenized_dialog += tokenized_ids + [hparams.sep_id]
        inputs.append(tokenized_dialog)

    print('prediction...')

    # checkpoint = torch.load(pretrained_model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    pred_list = []
    for i_test in range(len(inputs) // hparams.batch_size):
        batch = get_batch(inputs, hparams.batch_size, i_test)
        logits = model(batch)[:, :-1]  # trim the OOD column
        _, pred = torch.max(logits, dim=1)
        pred_list += pred.tolist()
    assert sum(inputs, []).count(102) == len(pred_list)  # n_utter == n_pred

    print('labeling...')
    index_to_emotion = {0: 'neutral', 1: 'joy', 2: 'sadness', 3: 'anger'}
    for dialog in json_data:
        for utter_dict in dialog:
            utter_dict['emotion'] = index_to_emotion[pred_list[0]]
            pred_list.pop(0)
    # result = [{'name': name, 'email': email}, json_data]
    json.dump(json_data, open(hparams.train_dir+'/result.json', 'w'), indent=4, sort_keys=True)

# if __name__ == '__main__':
#     label('data/friends_dev.json',
#           None, None, None
#           # 'Kisu Yang', 'willow4@korea.ac.kr'
#           )
