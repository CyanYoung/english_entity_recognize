import json
import pickle as pk

from sklearn_crfsuite.metrics import flat_f1_score, flat_accuracy_score

from recognize import crf_predict, dnn_predict, rnn_predict

from util import map_item


path_sent = 'data/test.json'
path_label_ind = 'feat/nn/label_ind.pkl'
with open(path_sent, 'r') as f:
    sents = json.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

slots = list(label_inds.keys())
slots.remove('N')
slots.remove('O')

funcs = {'crf': crf_predict,
         'dnn': dnn_predict,
         'rnn': rnn_predict,
         'rnn_bi': rnn_predict,
         'rnn_bi_crf': rnn_predict}


def test(name, sents):
    predict = map_item(name, funcs)
    label_mat, pred_mat = list(), list()
    for text, quaples in sents.items():
        words = text.split()
        tags, labels = list(), list()
        for quaple in quaples:
            tags.append(quaple['pos'])
            labels.append(quaple['label'])
        label_mat.append(labels)
        if name == 'crf':
            pairs = predict(words, tags)
        else:
            pairs = predict(words, name)
        preds = [pred for word, pred in pairs]
        pred_mat.append(preds)
    f1 = flat_f1_score(label_mat, pred_mat, average='weighted', labels=slots)
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, flat_accuracy_score(label_mat, pred_mat)))


if __name__ == '__main__':
    test('crf', sents)
    test('dnn', sents)
    test('rnn', sents)
    test('rnn_bi', sents)
    test('rnn_bi_crf', sents)
