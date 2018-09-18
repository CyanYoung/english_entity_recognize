import pickle as pk

import numpy as np

import nltk
import enchant

from featurize import sent2feat

from build import nn_compile

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from util import map_pos, map_path, map_model


def load_nn_crf(name, embed_mat, seq_len, class_num, paths):
    model = nn_compile(name, embed_mat, seq_len, class_num)
    model.load_weights(map_path(name, paths))
    return model


wnl = nltk.WordNetLemmatizer()
ed = enchant.Dict('en_US')

path_crf = 'model/crf.pkl'
with open(path_crf, 'rb') as f:
    crf = pk.load(f)

win_dist = 2
seq_len = 150

path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/nn/embed.pkl'
path_label_ind = 'feat/nn/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = dict()
for label, ind in label_inds.items():
    ind_labels[ind] = label


paths = {'dnn': 'model/dnn.h5',
         'rnn': 'model/rnn.h5',
         'rnn_bi': 'model/rnn_bi.h5',
         'rnn_bi_crf': 'model/rnn_bi_crf.h5'}

models = {'dnn': load_model(map_path('dnn', paths)),
          'rnn': load_model(map_path('rnn', paths)),
          'rnn_bi': load_model(map_path('rnn_bi', paths)),
          'rnn_bi_crf': load_nn_crf('rnn_bi_crf', embed_mat, seq_len, len(label_inds), paths)}


def crf_predict(words, tags):
    quaples = list()
    for word, tag in zip(words, tags):
        quaple = dict()
        quaple['word'] = word
        quaple['pos'] = tag
        quaples.append(quaple)
    sent_feat = sent2feat(quaples)
    preds = crf.predict([sent_feat])[0]
    pairs = list()
    for word, pred in zip(words, preds):
        pairs.append((word, pred))
    return pairs


def dnn_predict(words, name):
    seq = word2ind.texts_to_sequences([' '.join(words)])[0]
    align_wins = list()
    win_len = win_dist * 2 + 1
    buf = list(np.zeros(win_dist, dtype=int))
    buf_seq = buf + seq + buf
    for u_bound in range(win_len, len(buf_seq) + 1):
        align_win = pad_sequences([buf_seq[:u_bound]], maxlen=win_len)[0]
        align_wins.append(align_win)
    align_wins = np.array(align_wins)
    model = map_model(name, models)
    probs = model.predict(align_wins)
    inds = np.argmax(probs, axis=1)
    preds = [ind_labels[ind] for ind in inds]
    pairs = list()
    for word, pred in zip(words, preds):
        pairs.append((word, pred))
    return pairs


def rnn_predict(words, name):
    seq = word2ind.texts_to_sequences([' '.join(words)])[0]
    align_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_model(name, models)
    probs = model.predict(align_seq)[0]
    inds = np.argmax(probs, axis=1)
    preds = [ind_labels[ind] for ind in inds[-len(words):]]
    pairs = list()
    for word, pred in zip(words, preds):
        pairs.append((word, pred))
    return pairs


if __name__ == '__main__':
    while True:
        text = input('text: ')
        words = nltk.word_tokenize(text)
        pairs = nltk.pos_tag(words)
        words = [wnl.lemmatize(word, map_pos(tag)) for word, tag in pairs]
        tags = [tag for word, tag in pairs]
        print('crf: %s' % crf_predict(words, tags))
        print('dnn: %s' % dnn_predict(words, 'dnn'))
        print('rnn: %s' % rnn_predict(words, 'rnn'))
        print('rnn_bi: %s' % rnn_predict(words, 'rnn_bi'))
        print('rnn_bi_crf: %s' % rnn_predict(words, 'rnn_bi_crf'))
