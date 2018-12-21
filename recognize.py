import pickle as pk

import numpy as np

import nltk

from featurize import sent2feat

from keras.models import Model, load_model
from keras.layers import Input, Embedding

from keras_contrib.layers import CRF as K_CRF

from keras.preprocessing.sequence import pad_sequences

from nn_arch import rnn_bi_crf

from util import map_pos, map_item


def define_nn_crf(embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    crf = K_CRF(class_num)
    output = rnn_bi_crf(embed_input, crf)
    return Model(input, output)


def load_nn_crf(name, embed_mat, seq_len, class_num, paths):
    model = define_nn_crf(embed_mat, seq_len, class_num)
    model.load_weights(map_item(name, paths))
    return model


def ind2label(label_inds):
    ind_labels = dict()
    for word, ind in label_inds.items():
        ind_labels[ind] = word
    return ind_labels


lemmatizer = nltk.WordNetLemmatizer()

path_crf = 'model/crf.pkl'
with open(path_crf, 'rb') as f:
    crf = pk.load(f)

win_dist = 3
seq_len = 200

path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/nn/embed.pkl'
path_label_ind = 'feat/nn/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

paths = {'dnn': 'model/dnn.h5',
         'rnn': 'model/rnn.h5',
         'rnn_bi': 'model/rnn_bi.h5',
         'rnn_bi_crf': 'model/rnn_bi_crf.h5'}

models = {'dnn': load_model(map_item('dnn', paths)),
          'rnn': load_model(map_item('rnn', paths)),
          'rnn_bi': load_model(map_item('rnn_bi', paths)),
          'rnn_bi_crf': load_nn_crf('rnn_bi_crf', embed_mat, seq_len, len(label_inds), paths)}


def clean(text):
    words = nltk.word_tokenize(text)
    pairs = nltk.pos_tag(words)
    words = [lemmatizer.lemmatize(word, map_pos(tag)) for word, tag in pairs]
    tags = [tag for word, tag in pairs]
    return words, tags


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
    trunc_wins = list()
    win_len = win_dist * 2 + 1
    buf = list(np.zeros(win_dist, dtype=int))
    buf_seq = buf + seq + buf
    for u_bound in range(win_len, len(buf_seq) + 1):
        trunc_win = pad_sequences([buf_seq[:u_bound]], maxlen=win_len)[0]
        trunc_wins.append(trunc_win)
    trunc_wins = np.array(trunc_wins)
    model = map_item(name, models)
    probs = model.predict(trunc_wins)
    inds = np.argmax(probs, axis=1)
    preds = [ind_labels[ind] for ind in inds]
    pairs = list()
    for word, pred in zip(words, preds):
        pairs.append((word, pred))
    return pairs


def rnn_predict(words, name):
    seq = word2ind.texts_to_sequences([' '.join(words)])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_item(name, models)
    probs = model.predict(pad_seq)[0]
    inds = np.argmax(probs, axis=1)
    preds = [ind_labels[ind] for ind in inds[-len(words):]]
    pairs = list()
    for word, pred in zip(words, preds):
        pairs.append((word, pred))
    return pairs


if __name__ == '__main__':
    while True:
        text = input('text: ')
        words, tags = clean(text)
        print('crf: %s' % crf_predict(words, tags))
        print('dnn: %s' % dnn_predict(words, 'dnn'))
        print('rnn: %s' % rnn_predict(words, 'rnn'))
        print('rnn_bi: %s' % rnn_predict(words, 'rnn_bi'))
        print('rnn_bi_crf: %s' % rnn_predict(words, 'rnn_bi_crf'))
