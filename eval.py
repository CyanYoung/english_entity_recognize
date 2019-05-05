import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from recognize import label_inds, ind_labels, predict

from util import map_item


path_test = 'data/test.json'
with open(path_test, 'r') as f:
    sents = json.load(f)

class_num = len(label_inds)

slots = list(ind_labels.keys())
slots.remove(label_inds['N'])
slots.remove(label_inds['O'])

paths = {'crf': 'metric/crf.csv',
         'dnn': 'metric/dnn.csv',
         'rnn': 'metric/rnn.csv',
         'rnn_crf': 'metric/rnn_crf.csv'}


def test(name, sents):
    flat_labels, flat_preds = [0], [0]
    for text, pairs in sents.items():
        labels = list()
        for pair in pairs:
            labels.append(label_inds[pair['label']])
        flat_labels.extend(labels)
        flat_preds.extend(predict(text, name))
    precs = precision_score(flat_labels, flat_preds, average=None)
    recs = recall_score(flat_labels, flat_preds, average=None)
    with open(map_item(name, paths), 'w') as f:
        f.write('label,prec,rec' + '\n')
        for i in range(1, class_num):
            f.write('%s,%.2f,%.2f\n' % (ind_labels[i], precs[i], recs[i]))
    f1 = f1_score(flat_labels, flat_preds, average='weighted', labels=slots)
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, accuracy_score(flat_labels, flat_preds)))


if __name__ == '__main__':
    test('crf', sents)
    test('dnn', sents)
    test('rnn', sents)
    test('rnn_crf', sents)
