import pickle as pk

import numpy as np

import nltk

from featurize import sent2feat

from keras.models import Model, load_model
from keras.layers import Input, Embedding

from keras_contrib.layers import CRF as K_CRF

from keras.preprocessing.sequence import pad_sequences

from nn_arch import rnn_crf

from util import map_pos, map_item


def define_nn_crf(embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    crf = K_CRF(class_num)
    output = rnn_crf(embed_input, crf)
    return Model(input, output)


def load_nn_crf(name, embed_mat, seq_len, class_num, paths):
    model = define_nn_crf(embed_mat, seq_len, class_num)
    model.load_weights(map_item(name, paths))
    return model


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


def crf_predict(words, tags, name):
    quaples = list()
    for word, tag in zip(words, tags):
        quaple = dict()
        quaple['word'] = word
        quaple['pos'] = tag
        quaples.append(quaple)
    sent = sent2feat(quaples)
    model = map_item(name, models)
    preds = model.predict([sent])[0]
    inds = list()
    for pred in preds:
        inds.append(label_inds[pred])
    return inds


def dnn_predict(words, name):
    seq = word2ind.texts_to_sequences([' '.join(words)])[0]
    trunc_wins = list()
    buf = [0] * int((win_len - 1) / 2)
    buf_seq = buf + seq + buf
    for u_bound in range(win_len, len(buf_seq) + 1):
        l_bound = u_bound - win_len
        trunc_wins.append(buf_seq[l_bound:u_bound])
    trunc_wins = np.array(trunc_wins)
    model = map_item(name, models)
    probs = model.predict(trunc_wins)
    return np.argmax(probs, axis=1)


def rnn_predict(words, name):
    seq = word2ind.texts_to_sequences([' '.join(words)])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_item(name, models)
    probs = model.predict(pad_seq)[0]
    bound = min(len(words), seq_len)
    return np.argmax(probs, axis=1)[-bound:]


lemmatizer = nltk.WordNetLemmatizer()

path_crf = 'model/crf.pkl'
with open(path_crf, 'rb') as f:
    crf = pk.load(f)

win_len = 7
seq_len = 200

path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/nn/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

funcs = {'crf': crf_predict,
         'dnn': dnn_predict,
         'rnn': rnn_predict}

paths = {'dnn': 'model/dnn.h5',
         'rnn': 'model/rnn.h5',
         'rnn_crf': 'model/rnn_crf.h5'}

models = {'crf': crf,
          'dnn': load_model(map_item('dnn', paths)),
          'rnn': load_model(map_item('rnn', paths)),
          'rnn_crf': load_nn_crf('rnn_crf', embed_mat, seq_len, len(label_inds), paths)}


def clean(text):
    words = nltk.word_tokenize(text)
    pairs = nltk.pos_tag(words)
    words = [lemmatizer.lemmatize(word, map_pos(tag)) for word, tag in pairs]
    tags = [tag for word, tag in pairs]
    return words, tags


def predict(text, name):
    words, tags = clean(text)
    func = map_item(name[:3], funcs)
    if name == 'crf':
        preds = func(words, tags, name)
    else:
        preds = func(words, name)
    if __name__ == '__main__':
        pairs = list()
        for word, pred in zip(words, preds):
            pairs.append((word, ind_labels[pred]))
        return pairs
    else:
        return preds


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('crf: %s' % predict(text, 'crf'))
        print('dnn: %s' % predict(text, 'dnn'))
        print('rnn: %s' % predict(text, 'rnn'))
        print('rnn_crf: %s' % predict(text, 'rnn_crf'))
