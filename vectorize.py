import json
import pickle as pk

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from util import sent2label


embed_len = 200
max_vocab = 10000
win_len = 7
seq_len = 200

path_word_vec = 'feat/nn/word_vec.pkl'
path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/nn/embed.pkl'
path_label_ind = 'feat/nn/label_ind.pkl'


def embed(sents, path_word2ind, path_word_vec, path_embed):
    texts = sents.keys()
    model = Tokenizer(num_words=max_vocab, filters='', lower=True, oov_token='oov')
    model.fit_on_texts(texts)
    word_inds = model.word_index
    with open(path_word2ind, 'wb') as f:
        pk.dump(model, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.keys()
    vocab_num = min(max_vocab + 1, len(word_inds) + 1)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def label2ind(sents, path_label_ind):
    labels = list()
    for quaples in sents.values():
        labels.extend(sent2label(quaples))
    labels = sorted(list(set(labels)))
    label_inds = dict()
    label_inds['N'] = 0
    for i in range(len(labels)):
        label_inds[labels[i]] = i + 1
    with open(path_label_ind, 'wb') as f:
        pk.dump(label_inds, f)


def align_dnn(sents, path_sent, path_label):
    texts = sents.keys()
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    seqs = model.texts_to_sequences(texts)
    trunc_wins = list()
    buf = [0] * int((win_len - 1) / 2)
    for seq in seqs:
        buf_seq = buf + seq + buf
        for u_bound in range(win_len, len(buf_seq) + 1):
            l_bound = u_bound - win_len
            trunc_wins.append(buf_seq[l_bound:u_bound])
    trunc_wins = np.array(trunc_wins)
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    class_num = len(label_inds)
    inds = list()
    for quaples in sents.values():
        for quaple in quaples:
            inds.append(label_inds[quaple['label']])
    inds = to_categorical(inds, num_classes=class_num)
    with open(path_sent, 'wb') as f:
        pk.dump(trunc_wins, f)
    with open(path_label, 'wb') as f:
        pk.dump(inds, f)


def align_rnn(sents, path_sent, path_label):
    texts = sents.keys()
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    seqs = model.texts_to_sequences(texts)
    pad_seqs = pad_sequences(seqs, maxlen=seq_len)
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    class_num = len(label_inds)
    ind_mat = list()
    for quaples in sents.values():
        inds = list()
        for quaple in quaples:
            inds.append(label_inds[quaple['label']])
        pad_inds = pad_sequences([inds], maxlen=seq_len)[0]
        pad_inds = to_categorical(pad_inds, num_classes=class_num)
        ind_mat.append(pad_inds)
    ind_mat = np.array(ind_mat)
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)
    with open(path_label, 'wb') as f:
        pk.dump(ind_mat, f)


def vectorize(paths, mode):
    with open(paths['data'], 'r') as f:
        sents = json.load(f)
    if mode == 'train':
        embed(sents, path_word2ind, path_word_vec, path_embed)
        label2ind(sents, path_label_ind)
    align_dnn(sents, paths['dnn_sent'], paths['dnn_label'])
    align_rnn(sents, paths['rnn_sent'], paths['rnn_label'])


if __name__ == '__main__':
    paths = dict()
    prefix = 'feat/nn/'
    paths['data'] = 'data/train.json'
    paths['dnn_sent'] = prefix + 'dnn_sent_train.pkl'
    paths['dnn_label'] = prefix + 'dnn_label_train.pkl'
    paths['rnn_sent'] = prefix + 'rnn_sent_train.pkl'
    paths['rnn_label'] = prefix + 'rnn_label_train.pkl'
    vectorize(paths, 'train')
    paths['data'] = 'data/dev.json'
    paths['dnn_sent'] = prefix + 'dnn_sent_dev.pkl'
    paths['dnn_label'] = prefix + 'dnn_label_dev.pkl'
    paths['rnn_sent'] = prefix + 'rnn_sent_dev.pkl'
    paths['rnn_label'] = prefix + 'rnn_label_dev.pkl'
    vectorize(paths, 'dev')
