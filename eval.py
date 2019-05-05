import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from recognize import label_inds, ind_labels, funcs

from util import map_item


path_test = 'data/test.json'
with open(path_test, 'r') as f:
    sents = json.load(f)

class_num = len(label_inds)

full_slots = list(ind_labels.keys())

part_slots = full_slots[:]
part_slots.remove(label_inds['N'])
part_slots.remove(label_inds['O'])

paths = {'crf': 'metric/crf.csv',
         'dnn': 'metric/dnn.csv',
         'rnn': 'metric/rnn.csv',
         'rnn_crf': 'metric/rnn_crf.csv'}


def test(name, sents):
    func = map_item(name[:3], funcs)
    flat_labels, flat_preds = [0], [0]
    for quaples in sents.values():
        words, tags, labels = list(), list(), list()
        for quaple in quaples:
            words.append(quaple['word'])
            tags.append(quaple['pos'])
            labels.append(label_inds[quaple['label']])
        flat_labels.extend(labels)
        if name == 'crf':
            preds = func(words, tags, name)
        else:
            preds = func(words, name)
        flat_preds.extend(preds)
    precs = precision_score(flat_labels, flat_preds, average=None, labels=full_slots)
    recs = recall_score(flat_labels, flat_preds, average=None, labels=full_slots)
    with open(map_item(name, paths), 'w') as f:
        f.write('label,prec,rec' + '\n')
        for i in range(1, class_num):
            f.write('%s,%.2f,%.2f\n' % (ind_labels[i], precs[i], recs[i]))
    f1 = f1_score(flat_labels, flat_preds, average='weighted', labels=part_slots)
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, accuracy_score(flat_labels, flat_preds)))


if __name__ == '__main__':
    test('crf', sents)
    test('dnn', sents)
    test('rnn', sents)
    test('rnn_crf', sents)
