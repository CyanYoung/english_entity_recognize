from keras.layers import Dense, LSTM, Dropout, Flatten, Bidirectional


def dnn(embed_input, class_num):
    da1 = Dense(500, activation='relu')
    da2 = Dense(200, activation='relu')
    da3 = Dense(class_num, activation='softmax')
    x = Flatten()(embed_input)
    x = da1(x)
    x = da2(x)
    x = Dropout(0.5)(x)
    return da3(x)


def rnn(embed_input, class_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    da = Dense(class_num, activation='softmax')
    x = ra(embed_input)
    x = Dropout(0.5)(x)
    return da(x)


def rnn_bi(embed_input, class_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(class_num, activation='softmax')
    x = ba(embed_input)
    x = Dropout(0.5)(x)
    return da(x)


def rnn_bi_crf(embed_input, crf):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    x = ba(embed_input)
    x = Dropout(0.5)(x)
    return crf(x)
