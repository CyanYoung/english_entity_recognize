## English Entity Recognize 2018-9

#### 1.preprocess

prepare() 将每行数据处理为 (word, pos, chunk, label) 的四元组

map_pos() 将标准词性映射为 lemmatize() 可识别的，enchant 检查拼写错误

#### 2.explore

统计词汇、长度、实体的频率，条形图可视化，计算 slot_per_sent 指标

#### 3.featurize

ml 特征化，sent2feat() 将词转为小写后，增加是否句首、句尾等特征

#### 4.vectorize

nn 向量化，label2ind() 增设标签 N，trunc() 为 dnn 截取、pad() 为 rnn 填充

#### 5.build

crf_fit() 通过 crf，nn_fit() 通过 dnn、rnn、rnn_crf 构建实体识别模型

#### 6.recognize

rnn_crf 无法 load_model()、定义后 load_weights()，word_tokenize() 分词

pos_tag() 词性标注、lemmatize() 词形还原，每句返回 (word, pred) 的二元组