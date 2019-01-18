import json
import pickle as pk

from random import shuffle

from sklearn_crfsuite import CRF as S_CRF

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from keras_contrib.layers import CRF as K_CRF

from nn_arch import dnn, rnn, rnn_bi, rnn_bi_crf

from util import map_item


min_freq = 5
batch_size = 128

path_embed = 'feat/nn/embed.pkl'
path_label_ind = 'feat/nn/label_ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

class_num = len(label_inds)

funcs = {'dnn': dnn,
         'rnn': rnn,
         'rnn_bi': rnn_bi,
         'rnn_bi_crf': rnn_bi_crf}

paths = {'dnn': 'model/dnn.h5',
         'rnn': 'model/rnn.h5',
         'rnn_bi': 'model/rnn_bi.h5',
         'rnn_bi_crf': 'model/rnn_bi_crf.h5',
         'dnn_plot': 'model/plot/dnn.png',
         'rnn_plot': 'model/plot/rnn.png',
         'rnn_bi_plot': 'model/plot/rnn_bi.png',
         'rnn_bi_crf_plot': 'model/plot/rnn_bi_crf.png'}


def crf_fit(path_sent, path_label, path_crf):
    with open(path_sent, 'r') as f:
        sent_feats = json.load(f)
    with open(path_label, 'r') as f:
        labels = json.load(f)
    sents_labels = list(zip(sent_feats, labels))
    shuffle(sents_labels)
    sent_feats, labels = zip(*sents_labels)
    crf = S_CRF(algorithm='lbfgs', min_freq=min_freq, c1=0.1, c2=0.1,
                max_iterations=100, all_possible_transitions=True)
    crf.fit(sent_feats, labels)
    with open(path_crf, 'wb') as f:
        pk.dump(crf, f)


def nn_load(path_feats):
    with open(path_feats['sent_train'], 'rb') as f:
        train_sents = pk.load(f)
    with open(path_feats['label_train'], 'rb') as f:
        train_labels = pk.load(f)
    with open(path_feats['sent_dev'], 'rb') as f:
        dev_sents = pk.load(f)
    with open(path_feats['label_dev'], 'rb') as f:
        dev_labels = pk.load(f)
    return train_sents, train_labels, dev_sents, dev_labels


def nn_compile(name, embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    if name == 'rnn_bi_crf':
        crf = K_CRF(class_num)
        output = func(embed_input, crf)
        loss, acc = crf.loss_function, crf.accuracy
    else:
        output = func(embed_input, class_num)
        loss, acc = 'categorical_crossentropy', 'accuracy'
    model = Model(input, output)
    model.summary()
    plot_model(model, map_item(name + '_plot', paths), show_shapes=True)
    model.compile(loss=loss, optimizer=Adam(lr=0.001), metrics=[acc])
    return model


def nn_fit(name, epoch, embed_mat, class_num, path_feats):
    train_sents, train_labels, dev_sents, dev_labels = nn_load(path_feats)
    seq_len = len(train_sents[0])
    model = nn_compile(name, embed_mat, seq_len, class_num)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit(train_sents, train_labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_data=(dev_sents, dev_labels))


if __name__ == '__main__':
    prefix = 'feat/crf/'
    path_sent = prefix + 'sent_train.json'
    path_label = prefix + 'label_train.json'
    path_crf = 'model/crf.pkl'
    crf_fit(path_sent, path_label, path_crf)
    path_feats = dict()
    prefix = 'feat/nn/'
    path_feats['sent_train'] = prefix + 'dnn_sent_train.pkl'
    path_feats['label_train'] = prefix + 'dnn_label_train.pkl'
    path_feats['sent_dev'] = prefix + 'dnn_sent_dev.pkl'
    path_feats['label_dev'] = prefix + 'dnn_label_dev.pkl'
    nn_fit('dnn', 10, embed_mat, class_num, path_feats)
    path_feats['sent_train'] = prefix + 'rnn_sent_train.pkl'
    path_feats['label_train'] = prefix + 'rnn_label_train.pkl'
    path_feats['sent_dev'] = prefix + 'rnn_sent_dev.pkl'
    path_feats['label_dev'] = prefix + 'rnn_label_dev.pkl'
    nn_fit('rnn', 10, embed_mat, class_num, path_feats)
    nn_fit('rnn_bi', 10, embed_mat, class_num, path_feats)
    nn_fit('rnn_bi_crf', 10, embed_mat, class_num, path_feats)
