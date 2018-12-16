## English Entity Recognize 2018-9

#### 1.preprocess

prepare() 将每行数据分割为 (word, pos, chunk, label) 四元组

保存为 text 索引的字典，map_pos() 将标准词性映射为 lemmatize() 可识别的

当 word 全字母、全小写时，使用 enchant 检查拼写错误

#### 2.explore

统计词汇、长度、实体的频率，条形图可视化，计算 slot_per_sent 指标

#### 3.featurize

crf 特征化，sent2feat() 将词转为小写后，增加是否句首、句尾等特征

#### 4.vectorize

nn 向量化，label2ind() 增设标签 N，trunc() 为 dnn 截取、pad() 为 rnn 填充

#### 5.build

crf_fit() 通过 crf 构建实体识别模型，使用 L1 和 L2 正则化

nn_fit() 分别通过 dnn、rnn、rnn_bi、rnn_bi_crf，train 80% / dev 20% 划分

#### 6.recognize

word_tokenize() 分词、pos_tag() 词性标注、lemmatize() 词形还原

rnn_bi_crf  无法使用 load_model()、先 nn_compile() 再 load_weights()

#### 7.eval

分别调用 crf_predict()、dnn_predict()、rnn_predict()，计算准确率、f1 值
