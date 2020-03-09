import os
import json
import pickle
import torch

from EmotionX_KU_master.hparams import EMOTIONX_MODEL_HPARAMS
from pytorch_pretrained_bert import BertTokenizer
from EmotionX_KU_master.models import EmotionX_Model
from EmotionX_KU_master.utils import get_batch


def label(test_json_path):
    if not torch.cuda.is_available():
        raise NotImplementedError()
    hparams = type('', (object,), EMOTIONX_MODEL_HPARAMS)()  # dict to class
    hparams.n_appear = pickle.load(open(hparams.save_dir + 'hparams.txt', 'rb'))  # dict to class

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

    model = EmotionX_Model(hparams)
    model.load_state_dict(torch.load(hparams.save_dir + hparams.trained_model))
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
    json.dump(json_data, open(hparams.train_dir + hparams.result, 'w'), indent=4, sort_keys=True)


if __name__ == '__main__':
    label('data/friends_dev.json')
