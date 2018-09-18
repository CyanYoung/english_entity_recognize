import json

from util import sent2label


def sent2feat(quaples):
    sent_feat = list()
    for i in range(len(quaples)):
        quaple = quaples[i]
        word = quaple['word']
        word_feat = {
            'lower': word.lower(),
            'isupper': word.isupper(),
            'istitle': word.istitle(),
            'isdigit': word.isdigit(),
            'suffix_2': word.lower()[-2:],
            'suffix_3': word.lower()[-3:],
            'pos': quaple['pos'],
        }
        if i > 0:
            last_quaple = quaples[i - 1]
            last_word = last_quaple['word']
            word_feat.update({
                'last_lower': last_word.lower(),
                'last_isupper': last_word.isupper(),
                'last_istitle': last_word.istitle(),
                'last_pos': last_quaple['pos'],
            })
        else:
            word_feat['bos'] = True
        if i < len(quaples) - 1:
            next_quaple = quaples[i + 1]
            next_word = next_quaple['word']
            word_feat.update({
                'next_lower': next_word.lower(),
                'next_isupper': next_word.isupper(),
                'next_istitle': next_word.istitle(),
                'next_pos': next_quaple['pos'],
            })
        else:
            word_feat['eos'] = True
        sent_feat.append(word_feat)
    return sent_feat


def featurize(path_data, path_sent, path_label):
    with open(path_data, 'r') as f:
        sents = json.load(f)
    sent_feats = list()
    labels = list()
    for quaples in sents.values():
        sent_feats.append(sent2feat(quaples))
        labels.append(sent2label(quaples))
    with open(path_sent, 'w') as f:
        json.dump(sent_feats, f, ensure_ascii=False, indent=4)
    with open(path_label, 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    path_data = 'data/train.json'
    path_sent = 'feat/crf/sent_train.json'
    path_label = 'feat/crf/label_train.json'
    featurize(path_data, path_sent, path_label)
    path_data = 'data/dev.json'
    path_sent = 'feat/crf/sent_dev.json'
    path_label = 'feat/crf/label_dev.json'
    featurize(path_data, path_sent, path_label)
